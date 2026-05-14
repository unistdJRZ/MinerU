import asyncio
import glob
import hashlib
import json
import os
import re
import uuid
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_markdown
from mineru.backend.vlm.vlm_middle_json_mkcontent import mk_blocks_to_markdown
from mineru.cli.common import aio_do_parse, do_parse, image_suffixes, pdf_suffixes, read_fn
from mineru.cli.job_store import JobStore
from mineru.utils.cli_parser import arg_parse
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.version import __version__

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_FINISHED = "FINISHED"
STATUS_FAIL = "FAIL"
DEFAULT_MAX_CONCURRENT_JOBS = 1
DEFAULT_JOB_STORE_PATH = os.path.join(".", "output", "mineru_jobs.sqlite3")
CACHE_SCHEMA_VERSION = 1
SERVER_CONFIG_KEYS = {"max_concurrent_jobs", "job_store_path"}


def get_ocr_jobs() -> Dict[str, Dict[str, Any]]:
    if not hasattr(app.state, "ocr_jobs"):
        app.state.ocr_jobs = {}
    return app.state.ocr_jobs


def get_ocr_tasks() -> Dict[str, asyncio.Task]:
    if not hasattr(app.state, "ocr_tasks"):
        app.state.ocr_tasks = {}
    return app.state.ocr_tasks


def get_max_concurrent_jobs() -> int:
    raw_config = getattr(app.state, "config", {})
    raw_limit = raw_config.get("max_concurrent_jobs", DEFAULT_MAX_CONCURRENT_JOBS)

    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid max_concurrent_jobs={raw_limit!r}, fallback to {DEFAULT_MAX_CONCURRENT_JOBS}"
        )
        return DEFAULT_MAX_CONCURRENT_JOBS

    if limit < 1:
        logger.warning(
            f"max_concurrent_jobs must be >= 1, got {limit!r}. "
            f"Fallback to {DEFAULT_MAX_CONCURRENT_JOBS}"
        )
        return DEFAULT_MAX_CONCURRENT_JOBS

    return limit


def get_parse_runtime_config() -> Dict[str, Any]:
    raw_config = dict(getattr(app.state, "config", {}))
    return {
        key: value
        for key, value in raw_config.items()
        if key not in SERVER_CONFIG_KEYS
    }


def get_job_semaphore() -> asyncio.Semaphore:
    concurrency_limit = get_max_concurrent_jobs()
    current_limit = getattr(app.state, "ocr_job_semaphore_limit", None)

    if not hasattr(app.state, "ocr_job_semaphore") or current_limit != concurrency_limit:
        app.state.ocr_job_semaphore = asyncio.Semaphore(concurrency_limit)
        app.state.ocr_job_semaphore_limit = concurrency_limit

    return app.state.ocr_job_semaphore


def get_job_store_path() -> str:
    raw_config = getattr(app.state, "config", {})
    return str(raw_config.get("job_store_path", DEFAULT_JOB_STORE_PATH))


def get_job_store() -> JobStore:
    current_path = os.path.abspath(get_job_store_path())
    initialized_path = getattr(app.state, "job_store_path", None)

    if not hasattr(app.state, "job_store") or initialized_path != current_path:
        job_store = JobStore(current_path)
        job_store.mark_incomplete_jobs_failed(
            "Parse interrupted because the service restarted before completion"
        )
        app.state.job_store = job_store
        app.state.job_store_path = current_path

    return app.state.job_store


def bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_sha256(payload: Dict[str, Any]) -> str:
    payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def resolve_parse_dir(unique_dir: str, pdf_name: str, backend: str, parse_method: str) -> str:
    if backend.startswith("pipeline"):
        return os.path.join(unique_dir, pdf_name, parse_method)
    return os.path.join(unique_dir, pdf_name, "vlm")


def can_rebuild_job_content(job: Dict[str, Any]) -> bool:
    for pdf_name in job["source_pdf_file_names"]:
        parse_dir = resolve_parse_dir(
            unique_dir=job["unique_dir"],
            pdf_name=pdf_name,
            backend=job["backend"],
            parse_method=job["parse_method"],
        )
        if not os.path.exists(parse_dir):
            return False
    return True


def build_result_hash(
    *,
    pdf_hashes: List[str],
    actual_lang_list: List[str],
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: Optional[str],
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    start_page_id: int,
    end_page_id: int,
    config: Dict[str, Any],
) -> str:
    return json_sha256(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "mineru_version": __version__,
            "pdf_hashes": pdf_hashes,
            "lang_list": actual_lang_list,
            "backend": backend,
            "parse_method": parse_method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "server_url": server_url,
            "return_md": return_md,
            "return_middle_json": return_middle_json,
            "return_model_output": return_model_output,
            "return_content_list": return_content_list,
            "return_images": return_images,
            "start_page_id": start_page_id,
            "end_page_id": end_page_id,
            "config": config,
        }
    )


def sanitize_filename(filename: str) -> str:
    """
    格式化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._-
    禁止隐藏文件
    """
    sanitized = re.sub(r"[/\\\.]{2,}|[/\\]", "", filename)
    sanitized = re.sub(r"[^\w.-]", "_", sanitized, flags=re.UNICODE)
    if sanitized.startswith("."):
        sanitized = "_" + sanitized[1:]
    return sanitized or "unnamed"


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(
    file_suffix_identifier: str,
    pdf_name: str,
    parse_dir: str,
    parse_json: bool = False,
) -> Optional[Any]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            if parse_json:
                return json.load(fp)
            return fp.read()
    return None


def replace_image_with_base64(markdown_text: str, image_dir_path: str) -> str:
    if not markdown_text or not os.path.isdir(image_dir_path):
        return markdown_text

    pattern = r"!\[(?:[^\]]*)\]\(([^)]+)\)"

    def replace(match: re.Match[str]) -> str:
        relative_path = match.group(1)
        image_name = os.path.basename(relative_path)
        if not image_name.endswith(".jpg"):
            return match.group(0)

        full_path = os.path.join(image_dir_path, image_name)
        if not os.path.exists(full_path):
            return match.group(0)

        return f"![{image_name}](data:image/jpeg;base64,{encode_image(full_path)})"

    return re.sub(pattern, replace, markdown_text)


def middle_json_to_chunks(
    middle_json: Dict[str, Any],
    backend: str,
    formula_enable: bool,
    table_enable: bool,
    image_dir_path: str,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    pdf_info = middle_json.get("pdf_info", [])
    middle_backend = str(middle_json.get("_backend") or backend or "")
    is_pipeline_backend = middle_backend.startswith("pipeline")

    for page_idx, page_info in enumerate(pdf_info):
        para_blocks = page_info.get("para_blocks") or []
        page_size = page_info.get("page_size") or []
        page_index = page_info.get("page_idx", page_idx)

        for para_block in para_blocks:
            if is_pipeline_backend:
                md_list = make_blocks_to_markdown(
                    [para_block],
                    MakeMode.MM_MD,
                    img_buket_path="images",
                )
            else:
                md_list = mk_blocks_to_markdown(
                    [para_block],
                    MakeMode.MM_MD,
                    formula_enable,
                    table_enable,
                    img_buket_path="images",
                )

            if not md_list:
                continue

            markdown = replace_image_with_base64(md_list[0], image_dir_path)
            if not markdown:
                continue

            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "page_idx": page_index,
                    "page_size": page_size,
                    "bbox": para_block.get("bbox", []),
                    "markdown": markdown,
                    "content_type": para_block.get("type"),
                }
            )

    return chunks


async def load_uploaded_files(
    files: List[UploadFile],
    unique_dir: str,
) -> tuple[List[str], List[bytes], List[str]]:
    pdf_file_names = []
    pdf_bytes_list = []
    pdf_hashes = []

    for file in files:
        content = await file.read()
        file_path = Path(file.filename or f"{uuid.uuid4()}.pdf")

        temp_path = Path(unique_dir) / file_path.name
        with open(temp_path, "wb") as f:
            f.write(content)

        file_suffix = guess_suffix_by_path(temp_path)
        if file_suffix not in pdf_suffixes + image_suffixes:
            raise ValueError(f"Unsupported file type: {file_suffix}")

        pdf_bytes = read_fn(temp_path)
        pdf_bytes_list.append(pdf_bytes)
        pdf_hashes.append(bytes_sha256(pdf_bytes))
        pdf_file_names.append(file_path.stem)
        os.remove(temp_path)

    return pdf_file_names, pdf_bytes_list, pdf_hashes


def normalize_lang_list(lang_list: List[str], file_count: int) -> List[str]:
    if len(lang_list) == file_count:
        return lang_list
    return [lang_list[0] if lang_list else "ch"] * file_count


def build_json_result(
    unique_dir: str,
    source_pdf_file_names: List[str],
    response_pdf_file_names: Optional[List[str]],
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
) -> Dict[str, Any]:
    result_dict: Dict[str, Any] = {}
    response_names = response_pdf_file_names or source_pdf_file_names
    if len(response_names) != len(source_pdf_file_names):
        response_names = source_pdf_file_names

    for source_pdf_name, response_pdf_name in zip(source_pdf_file_names, response_names):
        result_dict[response_pdf_name] = {}
        data = result_dict[response_pdf_name]
        parse_dir = resolve_parse_dir(unique_dir, source_pdf_name, backend, parse_method)

        if os.path.exists(parse_dir):
            if return_md:
                data["md_content"] = get_infer_result(".md", source_pdf_name, parse_dir)
            if return_middle_json:
                middle_json = get_infer_result(
                    "_middle.json",
                    source_pdf_name,
                    parse_dir,
                    parse_json=True,
                )
                if middle_json is not None:
                    data["chunk"] = middle_json_to_chunks(
                        middle_json=middle_json,
                        backend=backend,
                        formula_enable=formula_enable,
                        table_enable=table_enable,
                        image_dir_path=os.path.join(parse_dir, "images"),
                    )
            if return_model_output:
                if backend.startswith("pipeline"):
                    data["model_output"] = get_infer_result("_model.json", source_pdf_name, parse_dir)
                else:
                    data["model_output"] = get_infer_result(
                        "_model_output.txt", source_pdf_name, parse_dir
                    )
            if return_content_list:
                data["content_list"] = get_infer_result(
                    "_content_list.json", source_pdf_name, parse_dir
                )
            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                image_paths = glob.glob(safe_pattern)
                data["images"] = {
                    os.path.basename(image_path): f"data:image/jpeg;base64,{encode_image(image_path)}"
                    for image_path in image_paths
                }

    return {
        "backend": backend,
        "version": __version__,
        "results": result_dict,
    }


def build_job_content(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if job["status"] == STATUS_FINISHED:
        if job.get("content") is not None:
            return job["content"]
        if can_rebuild_job_content(job):
            return build_json_result(
                unique_dir=job["unique_dir"],
                source_pdf_file_names=job["source_pdf_file_names"],
                response_pdf_file_names=job["response_pdf_file_names"],
                backend=job["backend"],
                parse_method=job["parse_method"],
                formula_enable=job["formula_enable"],
                table_enable=job["table_enable"],
                return_md=job["return_md"],
                return_middle_json=job["return_middle_json"],
                return_model_output=job["return_model_output"],
                return_content_list=job["return_content_list"],
                return_images=job["return_images"],
            )
        return {
            "error": "Persisted result files are missing and the response cannot be rebuilt"
        }

    if job["status"] == STATUS_FAIL:
        if job.get("content") is not None:
            return job["content"]
        if job.get("error_message"):
            return {"error": job["error_message"]}

    return None


async def run_parse_job(
    ocr_id: str,
    unique_dir: str,
    pdf_file_names: List[str],
    pdf_bytes_list: List[bytes],
    actual_lang_list: List[str],
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: Optional[str],
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    start_page_id: int,
    end_page_id: int,
    config: Dict[str, Any],
) -> None:
    jobs = get_ocr_jobs()
    job_store = get_job_store()
    semaphore = get_job_semaphore()

    async with semaphore:
        jobs[ocr_id] = {"status": STATUS_RUNNING, "content": None}
        job_store.update_job(ocr_id, status=STATUS_RUNNING)

        try:
            if backend == "vlm-vllm-async-engine":
                await aio_do_parse(
                    output_dir=unique_dir,
                    pdf_file_names=pdf_file_names,
                    pdf_bytes_list=pdf_bytes_list,
                    p_lang_list=actual_lang_list,
                    backend=backend,
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                    server_url=server_url,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_md=return_md,
                    f_dump_middle_json=return_middle_json,
                    f_dump_model_output=return_model_output,
                    f_dump_orig_pdf=False,
                    f_dump_content_list=return_content_list,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    **config,
                )
            else:
                await asyncio.to_thread(
                    do_parse,
                    output_dir=unique_dir,
                    pdf_file_names=pdf_file_names,
                    pdf_bytes_list=pdf_bytes_list,
                    p_lang_list=actual_lang_list,
                    backend=backend,
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                    server_url=server_url,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_md=return_md,
                    f_dump_middle_json=return_middle_json,
                    f_dump_model_output=return_model_output,
                    f_dump_orig_pdf=False,
                    f_dump_content_list=return_content_list,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    **config,
                )

            jobs[ocr_id] = {
                "status": STATUS_FINISHED,
                "content": build_json_result(
                    unique_dir=unique_dir,
                    source_pdf_file_names=pdf_file_names,
                    response_pdf_file_names=pdf_file_names,
                    backend=backend,
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                    return_md=return_md,
                    return_middle_json=return_middle_json,
                    return_model_output=return_model_output,
                    return_content_list=return_content_list,
                    return_images=return_images,
                ),
            }
            job_store.update_job(ocr_id, status=STATUS_FINISHED)
        except Exception as e:
            logger.exception(e)
            error_content = {"error": f"Failed to process file: {str(e)}"}
            jobs[ocr_id] = {
                "status": STATUS_FAIL,
                "content": error_content,
            }
            job_store.update_job(
                ocr_id,
                status=STATUS_FAIL,
                content=error_content,
                error_message=error_content["error"],
            )


@app.post(path="/file_parse")
async def parse_pdf(
    files: List[UploadFile] = File(...),
    ocr_id: Optional[str] = Form(None),
    output_dir: str = Form("./output"),
    lang_list: List[str] = Form(["ch"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    server_url: Optional[str] = Form(None),
    return_md: bool = Form(True),
    return_middle_json: bool = Form(True),
    return_model_output: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
    response_format_zip: bool = Form(False),
    start_page_id: int = Form(0),
    end_page_id: int = Form(99999),
):
    config = get_parse_runtime_config()
    job_store = get_job_store()

    try:
        if ocr_id is not None:
            ocr_id = ocr_id.strip()
            if not ocr_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "ocr_id cannot be empty"},
                )

        if response_format_zip:
            return JSONResponse(
                status_code=400,
                content={"error": "response_format_zip is not supported in async mode"},
            )

        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        pdf_file_names, pdf_bytes_list, pdf_hashes = await load_uploaded_files(files, unique_dir)
        actual_lang_list = normalize_lang_list(lang_list, len(pdf_file_names))
        result_hash = build_result_hash(
            pdf_hashes=pdf_hashes,
            actual_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            return_md=return_md,
            return_middle_json=return_middle_json,
            return_model_output=return_model_output,
            return_content_list=return_content_list,
            return_images=return_images,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            config=config,
        )

        cached_job = job_store.find_finished_job_by_result_hash(result_hash)
        if cached_job and can_rebuild_job_content(cached_job):
            resolved_ocr_id = ocr_id or str(uuid.uuid4())
            existing_task = get_ocr_tasks().get(resolved_ocr_id)
            if existing_task is not None and not existing_task.done():
                return JSONResponse(
                    status_code=409,
                    content={"error": f"ocr_id is already running: {resolved_ocr_id}"},
                )
            cached_record = {
                "ocr_id": resolved_ocr_id,
                "source_ocr_id": cached_job["ocr_id"],
                "result_hash": result_hash,
                "status": STATUS_FINISHED,
                "unique_dir": cached_job["unique_dir"],
                "output_dir": output_dir,
                "source_pdf_file_names": cached_job["source_pdf_file_names"],
                "response_pdf_file_names": pdf_file_names,
                "pdf_hashes": pdf_hashes,
                "lang_list": actual_lang_list,
                "config": config,
                "backend": backend,
                "parse_method": parse_method,
                "formula_enable": formula_enable,
                "table_enable": table_enable,
                "server_url": server_url,
                "return_md": return_md,
                "return_middle_json": return_middle_json,
                "return_model_output": return_model_output,
                "return_content_list": return_content_list,
                "return_images": return_images,
                "start_page_id": start_page_id,
                "end_page_id": end_page_id,
                "content": None,
                "error_message": None,
            }
            job_store.create_job(cached_record)
            get_ocr_jobs()[resolved_ocr_id] = {"status": STATUS_FINISHED, "content": None}
            try:
                os.rmdir(unique_dir)
            except OSError:
                pass
            return JSONResponse(status_code=200, content={"ocr_id": resolved_ocr_id})

        resolved_ocr_id = ocr_id or str(uuid.uuid4())
        existing_task = get_ocr_tasks().get(resolved_ocr_id)
        if existing_task is not None and not existing_task.done():
            return JSONResponse(
                status_code=409,
                content={"error": f"ocr_id is already running: {resolved_ocr_id}"},
            )

        jobs = get_ocr_jobs()
        jobs[resolved_ocr_id] = {"status": STATUS_PENDING, "content": None}
        job_store.create_job(
            {
                "ocr_id": resolved_ocr_id,
                "source_ocr_id": None,
                "result_hash": result_hash,
                "status": STATUS_PENDING,
                "unique_dir": unique_dir,
                "output_dir": output_dir,
                "source_pdf_file_names": pdf_file_names,
                "response_pdf_file_names": pdf_file_names,
                "pdf_hashes": pdf_hashes,
                "lang_list": actual_lang_list,
                "config": config,
                "backend": backend,
                "parse_method": parse_method,
                "formula_enable": formula_enable,
                "table_enable": table_enable,
                "server_url": server_url,
                "return_md": return_md,
                "return_middle_json": return_middle_json,
                "return_model_output": return_model_output,
                "return_content_list": return_content_list,
                "return_images": return_images,
                "start_page_id": start_page_id,
                "end_page_id": end_page_id,
                "content": None,
                "error_message": None,
            }
        )

        task = asyncio.create_task(
            run_parse_job(
                ocr_id=resolved_ocr_id,
                unique_dir=unique_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                actual_lang_list=actual_lang_list,
                backend=backend,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                server_url=server_url,
                return_md=return_md,
                return_middle_json=return_middle_json,
                return_model_output=return_model_output,
                return_content_list=return_content_list,
                return_images=return_images,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                config=config,
            )
        )

        tasks = get_ocr_tasks()
        tasks[resolved_ocr_id] = task
        task.add_done_callback(lambda _, job_id=resolved_ocr_id: get_ocr_tasks().pop(job_id, None))

        return JSONResponse(status_code=200, content={"ocr_id": resolved_ocr_id})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to submit parse task: {str(e)}"},
        )


@app.get(path="/content")
async def get_parse_content(ocr_id: str):
    job_info = get_job_store().get_job(ocr_id)
    memory_job_info = None
    if job_info is None:
        memory_job_info = get_ocr_jobs().get(ocr_id)
        job_info = memory_job_info

    if job_info is None:
        return JSONResponse(status_code=404, content={"error": f"ocr_id not found: {ocr_id}"})

    content = (
        build_job_content(job_info)
        if memory_job_info is None
        else memory_job_info.get("content")
    )
    status = job_info["status"]

    if (
        memory_job_info is None
        and status == STATUS_FINISHED
        and isinstance(content, dict)
        and "error" in content
    ):
        get_job_store().update_job(
            ocr_id,
            status=STATUS_FAIL,
            content=content,
            error_message=content["error"],
        )
        status = STATUS_FAIL

    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "content": content,
        },
    )


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):
    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs
    get_job_store()

    """启动MinerU FastAPI服务的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "mineru.cli.fast_api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()

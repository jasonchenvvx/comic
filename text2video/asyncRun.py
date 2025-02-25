import multiprocessing
import os
import re
import comic


FILE_PATH = "resources/text_file.txt"

def process_single_file(file_path: str, num: str):
    """示例处理函数（根据实际需求修改）"""
    with open(file_path, "r", encoding="utf-8") as f:
        sample_text = f.read()
    generate_video(sample_text, f"comic_video_{num}.mp4")
    return len(sample_text)

def split_large_file(file_path: str, output_dir: str, max_words=2000) -> List[str]:
    """分割大文件为多个小文件[7]"""
    chunks = []
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        words = f.read().split()

    for i in range(0, len(words), max_words):
        chunk = words[i:i+max_words]
        output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_part{i//max_words}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(" ".join(chunk))
        chunks.append(output_path)
    return chunks

def parallel_job_processor(input_dir: str, output_dir: str, workers=4):
    """并行任务处理框架[2]"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 分割文件
    all_chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            all_chunks.extend(split_large_file(file_path, output_dir))

    # 2. 并行处理
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_single_file, all_chunks)

    return results

if __name__ == "__main__":
    results = parallel_job_processor(
        input_dir="/Users/jason/IdeaProjects/ComicGenerator",
        output_dir="processed_chunks",
        workers=os.cpu_count()
    )
    print(f"处理完成，共处理 {len(results)} 个文件块")
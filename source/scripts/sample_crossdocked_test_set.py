import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count

def run_single_task(data_id):
    python_exec = sys.executable
    script_path = "./source/scripts/sample_diffusion_with_pocket.py"
    base_result_path = "./outputs_test_EGPO_w_sa/"
    os.makedirs(base_result_path, exist_ok=True)
    
    cmd = [
        python_exec,
        script_path,
        "--data_id", str(data_id),
        "--result_path", base_result_path
    ]
    
    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(data_id, "success")
        return (data_id, "success", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return (data_id, "fail", f"错误码：{e.returncode}，错误信息：{e.stderr.strip()}")
    except Exception as e:
        return (data_id, "error", str(e))

if __name__ == "__main__":
    # 使用CPU核心数的一半作为进程数（避免资源耗尽）
    process_num = max(1, cpu_count() // 2)
    print(f"使用 {process_num} 个进程并行执行...")
    
    # 创建进程池并执行
    with Pool(processes=process_num) as pool:
        results = pool.map(run_single_task, range(0, 100))
    
    # 统计结果
    success_count = 0
    fail_count = 0
    fail_ids = []
    for res in results:
        data_id, status, msg = res
        if status == "success":
            print(f"data_id={data_id} 执行成功：{msg}")
            success_count += 1
        else:
            print(f"data_id={data_id} 执行{status}：{msg}")
            fail_count += 1
            fail_ids.append(data_id)
    
    # 输出汇总
    print("\n========== 并行执行完成 ==========")
    print(f"总任务数：100")
    print(f"成功数：{success_count}")
    print(f"失败数：{fail_count}")
    if fail_ids:
        print(f"失败的data_id列表：{fail_ids}")
import os
import json
import http.client
import random
import time
import subprocess

# Get API key from environment variable
API_KEY = os.environ.get('GEMINI_API_KEY', '')
MODEL_NAME = "gemini-2.5-flash-preview-04-17-nothinking"

def upload_file_to_tmpfiles(file_path):
    """
    使用curl上传文件到tmpfiles.org并解析响应
    
    参数:
        file_path: 要上传的文件路径
    
    返回:
        解析后的JSON响应
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}
    
    # 构建curl命令
    curl_command = [
        "curl",
        "-F", f"file=@{file_path}",
        "https://tmpfiles.org/api/v1/upload"
    ]
    
    try:
        # 执行curl命令
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析JSON响应
        response = json.loads(result.stdout)
        
        # 提取重要信息
        if response.get("status") == "success":
            data = response.get("data", {})
            file_info = {
                "url": data.get("url"),
                "download_url": data.get("url", "").replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/"),
                "size": data.get("size"),
                "expires": data.get("expires")
            }
            return file_info
        else:
            return {"error": "上传失败", "response": response}
            
    except subprocess.CalledProcessError as e:
        return {"error": f"执行curl命令失败: {str(e)}", "stderr": e.stderr}
    except json.JSONDecodeError:
        return {"error": "无法解析JSON响应", "raw_response": result.stdout}
    except Exception as e:
        return {"error": f"发生未知错误: {str(e)}"}
    
def evaluation(
    input_data,
    modality,
    max_retries: int = 0,
    retry_delay: tuple = (1, 3),  # 重试延迟范围 (min, max) 秒
):
    """
    调用 API 并支持错误重试
    
    Args:
        input_data: 输入数据 (prompt, file_path, segments, _)
        modality: 模态类型（未使用，但保留参数）
        max_retries: 最大重试次数（默认 3）
        retry_delay: 重试延迟范围（秒），默认 (1, 3)
        
    Returns:
        str: API 返回的响应内容，失败时返回 None
    """
    prompt, file_path, segments, _ = input_data
    # video_info = upload_file_to_tmpfiles(file_path+'.mp4')
    file_name = os.path.basename(file_path+'.mp4')
    hf_file_path = "YOUR_HF_FILE_PATH"
    # print(video_info['download_url'])
    # 构建请求内容
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": hf_file_path}},
    ]
    payload = json.dumps({
        "model": MODEL_NAME,
        "stream": False,
        "messages": [{"role": "user", "content": content}],
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    # 重试逻辑
    for attempt in range(max_retries + 1):  # 最多尝试 max_retries + 1 次
        conn = None
        try:
            # 建立连接并发送请求
            conn = http.client.HTTPSConnection("api2.aigcbest.top")
            conn.request("POST", "/v1/chat/completions", payload, headers)
            
            # 获取响应
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            # print(data)
            
            # 检查 HTTP 状态码
            if res.status != 200:
                print(data)
                raise ValueError(f"API 返回错误状态码: {res.status}")
            
            # 解析 JSON
            response_data = json.loads(data)
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                print(data)
                print(hf_file_path)
                raise ValueError("API 返回无效数据: 缺少 'choices' 字段")
                
        except (http.client.HTTPException, json.JSONDecodeError, ValueError) as e:
            # 可预期的错误（网络问题、API 错误等）
            if attempt < max_retries:
                delay = random.uniform(*retry_delay)
                print(hf_file_path)
                print(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}，{delay:.1f} 秒后重试...")
                time.sleep(delay)
                continue
            else:
                print(f"所有重试失败，最终错误: {str(e)}")
                return None
                
        except Exception as e:
            # 其他未知错误
            print(f"未知错误: {str(e)}")
            return None
            
        finally:
            # 确保连接关闭
            if conn:
                conn.close()
    
    return None  # 所有重试失败
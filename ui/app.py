# -*- coding: utf-8 -*-
"""
ui/app.py

作者：Accilia
创建时间：2026-03-08
用途说明：
QA助手可视化界面后端API，处理用户请求并调用QA pipeline
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any, Generator

from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

from deepsea_qa.qa.pipeline import QAPipeline, QAPipelineConfig

# 设置环境变量
# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.api_keys import set_api_key_env
set_api_key_env()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'ui/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/qa', methods=['POST'])
def qa():
    """处理单个问题的请求"""
    data = request.json
    query = data.get('query', '').strip()
    llm_provider = data.get('llm_provider', 'zhipu')
    llm_model = data.get('llm_model', 'glm-4-plus')
    
    if not query:
        return jsonify({'error': '查询不能为空'}), 400
    
    try:
        cfg = QAPipelineConfig()
        pipe = QAPipeline(llm_provider=llm_provider, llm_model=llm_model, cfg=cfg)
        result = pipe.run(query)
        
        # 处理结果，提取需要的信息
        answer = result['generation'].get('answer', '')
        citations = result['generation'].get('citations', [])
        refs_gbt = result['generation'].get('refs_gbt', [])
        
        # 构建响应
        response = {
            'query': query,
            'answer': answer,
            'citations': citations,
            'refs_gbt': refs_gbt,
            'verified': result['generation'].get('verified', False)
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/qa/stream', methods=['POST'])
def qa_stream():
    """流式处理单个问题的请求"""
    data = request.json
    query = data.get('query', '').strip()
    llm_provider = data.get('llm_provider', 'zhipu')
    llm_model = data.get('llm_model', 'glm-4-plus')
    
    if not query:
        return jsonify({'error': '查询不能为空'}), 400
    
    def generate():
        try:
            cfg = QAPipelineConfig()
            # 注意：如果 QAPipeline 初始化很慢，也会导致前端超时或 abort
            pipe = QAPipeline(llm_provider=llm_provider, llm_model=llm_model, cfg=cfg)
            
            for event in pipe.stream(query):
                # 移除 chunk_id，避免在前端显示
                if 'chunk_id' in event:
                    del event['chunk_id']
                # 确保 json.dumps 不报错，使用 ensure_ascii=False 支持中文
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 捕获所有异常并发送给前端，防止连接静默断开
            print(f"Stream error: {str(e)}") # 后端打印日志以便调试
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    # 使用 flask.Response 包装生成器，这是标准做法
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no', # Nginx 特定头部，如果是直接运行 Flask 可忽略，但保留无害
            'Content-Type': 'text/event-stream' # 显式声明 Content-Type
        }
    )

@app.route('/api/qa/batch', methods=['POST'])
def qa_batch():
    """处理批量问题的请求"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取文件内容，解析问题列表
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            if not questions:
                return jsonify({'error': '文件中没有有效的问题'}), 400
            
            llm_provider = request.form.get('llm_provider', 'zhipu')
            llm_model = request.form.get('llm_model', 'glm-4-plus')
            
            # 处理每个问题
            results = []
            cfg = QAPipelineConfig()
            pipe = QAPipeline(llm_provider=llm_provider, llm_model=llm_model, cfg=cfg)
            
            for i, question in enumerate(questions):
                try:
                    result = pipe.run(question)
                    answer = result['generation'].get('answer', '')
                    citations = result['generation'].get('citations', [])
                    refs_gbt = result['generation'].get('refs_gbt', [])
                    
                    results.append({
                        'question': question,
                        'answer': answer,
                        'citations': citations,
                        'refs_gbt': refs_gbt,
                        'verified': result['generation'].get('verified', False)
                    })
                except Exception as e:
                    results.append({
                        'question': question,
                        'error': str(e)
                    })
            
            return jsonify({'questions': questions, 'results': results})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': '不支持的文件类型，仅支持txt文件'}), 400

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('ui/static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
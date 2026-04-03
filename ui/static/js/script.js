/**
 * script.js
 * 
 * 作者：Accilia
 * 创建时间：2026-03-08
 * 用途说明：
 * 前端交互逻辑，包括标签页切换、问题提交、流式处理、批量处理等
 */

// 标签页切换
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            
            // 更新标签按钮状态
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // 更新标签内容
            tabContents.forEach(content => {
                content.classList.add('hidden');
            });
            document.getElementById(tabId).classList.remove('hidden');
        });
    });
}

// 单个问题处理
function setupSingleQuery() {
    const submitButton = document.getElementById('submit-query');
    const queryInput = document.getElementById('query');
    const statusDiv = document.getElementById('status');
    const processStepsDiv = document.getElementById('process-steps');
    const answerDiv = document.getElementById('answer');
    const refsListDiv = document.getElementById('refs-list');
    
    submitButton.addEventListener('click', () => {
        const query = queryInput.value.trim();
        if (!query) {
            alert('请输入问题');
            return;
        }
        
        const llmProvider = document.getElementById('llm_provider').value;
        const llmModel = document.getElementById('llm_model').value;
        
        // 清空之前的结果
        answerDiv.innerHTML = '';
        refsListDiv.innerHTML = '';
        
        // 显示状态
        statusDiv.classList.remove('hidden');
        processStepsDiv.innerHTML = '<p>正在处理...</p>';
        
        // 流式处理请求
        let answer = '';
        let evidenceList = [];
        
        fetch('/api/qa/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                llm_provider: llmProvider,
                llm_model: llmModel
            })
        })
        .then(async response => {
            if (!response.ok) {
                throw new Error('网络响应错误');
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            function processChunk({ done, value }) {
                if (value) {
                    buffer += decoder.decode(value, { stream: true });
                }
                
                // 处理接收到的数据，按行分割
                const lines = buffer.split('\n');
                buffer = lines.pop(); // 保存最后一行（可能不完整）
                
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6);
                        if (dataStr) {
                            try {
                                const data = JSON.parse(dataStr);
                                
                                if (data.event === 'evidence') {
                                    // 处理证据
                                    evidenceList.push(data);
                                    processStepsDiv.innerHTML += `<p>📄 加载证据 ${data.idx}: ${data.title}</p>`;
                                } else if (data.event === 'answer_delta') {
                                    // 处理答案增量（草稿），显示在思考过程中
                                    answer += data.delta;
                                    // 检查是否已经有生成中的行
                                    let generatingLine = processStepsDiv.querySelector('.generating-line');
                                    if (!generatingLine) {
                                        generatingLine = document.createElement('p');
                                        generatingLine.className = 'generating-line';
                                        generatingLine.innerHTML = '💭 生成中：';
                                        processStepsDiv.appendChild(generatingLine);
                                    }
                                    // 在同一行内追加内容，过滤掉chunk_id信息和处理#号
                                    let filteredDelta = data.delta.replace(/\(chunk_id=[^)]*\)/g, '');
                                    // 将#号替换为更美观的标题格式
                                    filteredDelta = filteredDelta.replace(/#{1,6}\s+(.*?)(?=\n|$)/g, '<strong>$1</strong><br>');
                                    generatingLine.innerHTML += filteredDelta;
                                } else if (data.event === 'done') {
                                    // 草稿完成
                                    processStepsDiv.innerHTML += '<p>✏️ 生成草稿完成</p>';
                                    // 添加中间状态提示
                                    processStepsDiv.innerHTML += '<p>🔍 正在验证答案并生成最终结果...</p>';
                                } else if (data.event === 'final') {
                                    // 最终结果，显示在回答结果部分
                                    processStepsDiv.innerHTML += '<p>✅ 处理完成</p>';
                                    processStepsDiv.innerHTML += '<p>📋 正在格式化参考文献...</p>';
                                    // 延迟显示最终结果，让用户有时间看到处理过程
                                    setTimeout(() => {
                                        // 处理最终答案和参考文献
                                        const finalData = data.data;
                                        const generation = finalData.generation;
                                        
                                        // 显示最终答案，添加引用角标，过滤掉chunk_id信息
                                        let filteredAnswer = generation.answer.replace(/\(chunk_id=[^)]*\)/g, '');
                                        const formattedAnswer = addCitationMarkers(filteredAnswer, generation.citations);
                                        answerDiv.innerHTML = formattedAnswer;
                                        
                                        // 保存到历史记录
                                        saveToHistory(query, filteredAnswer);
                                        
                                        // 显示参考文献列表
                                        if (generation.refs_gbt && generation.refs_gbt.length > 0) {
                                            let refsHtml = '<ol>';
                                            generation.refs_gbt.forEach((ref, index) => {
                                                refsHtml += `<li>${ref}</li>`;
                                            });
                                            refsHtml += '</ol>';
                                            refsListDiv.innerHTML = refsHtml;
                                        } else {
                                            refsListDiv.innerHTML = '<p>无参考文献</p>';
                                        }
                                        
                                        // 5秒后隐藏状态
                                        setTimeout(() => {
                                            statusDiv.classList.add('hidden');
                                        }, 5000);
                                    }, 1000);
                                } else if (data.event === 'error') {
                                    // 错误处理
                                    processStepsDiv.innerHTML = `<p style="color: red;">错误：${data.message}</p>`;
                                }
                            } catch (e) {
                                console.error('解析事件数据失败:', e);
                            }
                        }
                    }
                });
                
                if (done) {
                    return;
                }
                
                return reader.read().then(processChunk);
            }
            
            const result_2 = await reader.read();
            return processChunk(result_2);
        })
        .catch(error => {
            console.error('请求失败:', error);
            processStepsDiv.innerHTML = `<p style="color: red;">错误：${error.message}</p>`;
            // 5秒后隐藏状态
            setTimeout(() => {
                statusDiv.classList.add('hidden');
            }, 5000);
        });
    });
}

// 批量问题处理
function setupBatchQuery() {
    const fileInput = document.getElementById('batch-file');
    const batchPreview = document.getElementById('batch-preview');
    const questionsList = document.getElementById('questions-list');
    const submitBatchButton = document.getElementById('submit-batch');
    const batchResultsContainer = document.getElementById('batch-results-container');
    
    // 文件预览
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                const questions = content.split('\n').filter(line => line.trim());
                
                if (questions.length > 0) {
                    batchPreview.classList.remove('hidden');
                    questionsList.innerHTML = '';
                    questions.forEach((q, index) => {
                        questionsList.innerHTML += `<p>${index + 1}. ${q}</p>`;
                    });
                } else {
                    batchPreview.classList.add('hidden');
                    alert('文件中没有有效的问题');
                }
            };
            reader.readAsText(file, 'utf-8');
        }
    });
    
    // 提交批量处理
    submitBatchButton.addEventListener('click', () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('请选择文件');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('llm_provider', document.getElementById('batch-llm_provider').value);
        formData.append('llm_model', document.getElementById('batch-llm_model').value);
        
        batchResultsContainer.innerHTML = '<p>正在处理，请稍候...</p>';
        
        fetch('/api/qa/batch', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                batchResultsContainer.innerHTML = `<p style="color: red;">错误：${data.error}</p>`;
                return;
            }
            
            batchResultsContainer.innerHTML = '';
            data.results.forEach((result, index) => {
                const resultItem = document.createElement('div');
                resultItem.className = 'batch-result-item';
                
                if (result.error) {
                    resultItem.innerHTML = `
                        <h4>问题 ${index + 1}: ${data.questions[index]}</h4>
                        <p style="color: red;">错误：${result.error}</p>
                    `;
                } else {
                    const formattedAnswer = addCitationMarkers(result.answer, result.citations);
                    let refsHtml = '';
                    if (result.refs_gbt && result.refs_gbt.length > 0) {
                        refsHtml = '<div class="batch-references"><h5>参考文献：</h5><ol>';
                        result.refs_gbt.forEach((ref, refIndex) => {
                            refsHtml += `<li>${ref}</li>`;
                        });
                        refsHtml += '</ol></div>';
                    }
                    
                    resultItem.innerHTML = `
                        <h4>问题 ${index + 1}: ${data.questions[index]}</h4>
                        <div class="batch-answer">${formattedAnswer}</div>
                        ${refsHtml}
                    `;
                }
                
                batchResultsContainer.appendChild(resultItem);
            });
        })
        .catch(error => {
            batchResultsContainer.innerHTML = `<p style="color: red;">请求失败：${error.message}</p>`;
        });
    });
}

// 添加引用角标
function addCitationMarkers(answer, citations) {
    if (!citations || citations.length === 0) {
        return answer;
    }
    
    let result = answer;
    let citationMap = new Map();
    
    // 构建引用映射
    citations.forEach((citation, index) => {
        const paperId = citation.paper_id;
        if (!citationMap.has(paperId)) {
            citationMap.set(paperId, index + 1);
        }
    });
    
    // 简单实现：在答案末尾添加引用标记
    // 注意：实际应用中可能需要更复杂的算法来匹配引用位置
    let refsUsed = new Set();
    citations.forEach(citation => {
        const paperId = citation.paper_id;
        const refNumber = citationMap.get(paperId);
        if (!refsUsed.has(refNumber)) {
            result += `<sup class="citation">[${refNumber}]</sup>`;
            refsUsed.add(refNumber);
        }
    });
    
    return result;
}

// 历史记录管理
function saveToHistory(query, answer) {
    // 获取现有历史记录
    let history = JSON.parse(localStorage.getItem('qa_history') || '[]');
    
    // 添加新记录
    const newRecord = {
        id: Date.now(),
        query: query,
        answer: answer,
        timestamp: new Date().toISOString()
    };
    
    // 限制历史记录数量为20条
    history.unshift(newRecord);
    if (history.length > 20) {
        history = history.slice(0, 20);
    }
    
    // 保存到localStorage
    localStorage.setItem('qa_history', JSON.stringify(history));
}

function loadHistory() {
    const historyList = document.getElementById('history-list');
    const history = JSON.parse(localStorage.getItem('qa_history') || '[]');
    
    if (history.length === 0) {
        historyList.innerHTML = '<p>暂无历史记录</p>';
        return;
    }
    
    historyList.innerHTML = '';
    history.forEach(record => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        // 格式化时间
        const date = new Date(record.timestamp);
        const formattedTime = date.toLocaleString();
        
        // 截取答案预览
        const answerPreview = record.answer.length > 100 ? record.answer.substring(0, 100) + '...' : record.answer;
        
        historyItem.innerHTML = `
            <div class="history-query">${record.query}</div>
            <div class="history-time">${formattedTime}</div>
            <div class="history-answer">${answerPreview}</div>
        `;
        
        // 点击历史记录重新加载问题
        historyItem.addEventListener('click', () => {
            // 切换到单个问题标签
            document.querySelector('[data-tab="single"]').click();
            // 填充问题
            document.getElementById('query').value = record.query;
        });
        
        historyList.appendChild(historyItem);
    });
}

function clearHistory() {
    if (confirm('确定要清空所有历史记录吗？')) {
        localStorage.removeItem('qa_history');
        loadHistory();
    }
}

// 模型和提供商的映射关系
const modelProviderMap = {
    // 提供商到默认模型的映射
    providerToModel: {
        'zhipu': 'glm-4-plus',
        'dashscope': 'qwen-plus',
        'openai': 'gpt-4o'
    },
    // 模型到提供商的映射
    modelToProvider: {
        'glm-4-plus': 'zhipu',
        'qwen-plus': 'dashscope',
        'gpt-4o': 'openai'
    }
};

// 设置模型和提供商的自动更改功能
function setupModelProviderSync() {
    // 单个问题的选择器
    const providerSelect = document.getElementById('llm_provider');
    const modelSelect = document.getElementById('llm_model');
    
    // 批量问题的选择器
    const batchProviderSelect = document.getElementById('batch-llm_provider');
    const batchModelSelect = document.getElementById('batch-llm_model');
    
    // 单个问题的提供商变化事件
    providerSelect.addEventListener('change', function() {
        const selectedProvider = this.value;
        const defaultModel = modelProviderMap.providerToModel[selectedProvider];
        if (defaultModel) {
            modelSelect.value = defaultModel;
        }
    });
    
    // 单个问题的模型变化事件
    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        const defaultProvider = modelProviderMap.modelToProvider[selectedModel];
        if (defaultProvider) {
            providerSelect.value = defaultProvider;
        }
    });
    
    // 批量问题的提供商变化事件
    batchProviderSelect.addEventListener('change', function() {
        const selectedProvider = this.value;
        const defaultModel = modelProviderMap.providerToModel[selectedProvider];
        if (defaultModel) {
            batchModelSelect.value = defaultModel;
        }
    });
    
    // 批量问题的模型变化事件
    batchModelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        const defaultProvider = modelProviderMap.modelToProvider[selectedModel];
        if (defaultProvider) {
            batchProviderSelect.value = defaultProvider;
        }
    });
}

// 初始化
function init() {
    setupTabs();
    setupSingleQuery();
    setupBatchQuery();
    setupModelProviderSync();
    
    // 初始化历史记录
    const historyTab = document.querySelector('[data-tab="history"]');
    historyTab.addEventListener('click', loadHistory);
    
    // 清空历史记录按钮
    const clearHistoryBtn = document.getElementById('clear-history');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearHistory);
    }
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', init);
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

function appendProcessStep(container, text, className = '') {
    const line = document.createElement('p');
    if (className) {
        line.className = className;
    }
    line.textContent = text;
    container.appendChild(line);
    return line;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function formatAnswerHtml(answer) {
    return escapeHtml(answer).replace(/\n/g, '<br>');
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
                                    appendProcessStep(processStepsDiv, `加载证据 ${data.idx}: ${data.title || '未命名文献'}`);
                                } else if (data.event === 'answer_delta') {
                                    let generatingLine = processStepsDiv.querySelector('.generating-line');
                                    if (!generatingLine) {
                                        generatingLine = appendProcessStep(processStepsDiv, '生成中：', 'generating-line');
                                    }
                                    const filteredDelta = String(data.delta || '').replace(/\(chunk_id=[^)]*\)/g, '');
                                    generatingLine.textContent += filteredDelta;
                                } else if (data.event === 'done') {
                                    appendProcessStep(processStepsDiv, '生成草稿完成');
                                    appendProcessStep(processStepsDiv, '正在验证答案并生成最终结果...');
                                } else if (data.event === 'final') {
                                    appendProcessStep(processStepsDiv, '处理完成');
                                    appendProcessStep(processStepsDiv, '正在格式化参考文献...');
                                    setTimeout(() => {
                                        const finalData = data.data;
                                        const generation = finalData.generation;
                                        
                                        let filteredAnswer = generation.answer.replace(/\(chunk_id=[^)]*\)/g, '');
                                        const formattedAnswer = addCitationMarkers(filteredAnswer, generation.citations);
                                        answerDiv.innerHTML = formattedAnswer;
                                        
                                        // 保存到历史记录
                                        saveToHistory(query, filteredAnswer);
                                        
                                        if (generation.refs_gbt && generation.refs_gbt.length > 0) {
                                            const refsOl = document.createElement('ol');
                                            generation.refs_gbt.forEach(ref => {
                                                const li = document.createElement('li');
                                                li.textContent = ref;
                                                refsOl.appendChild(li);
                                            });
                                            refsListDiv.replaceChildren(refsOl);
                                        } else {
                                            refsListDiv.textContent = '无参考文献';
                                        }
                                        
                                        setTimeout(() => {
                                            statusDiv.classList.add('hidden');
                                        }, 5000);
                                    }, 1000);
                                } else if (data.event === 'error') {
                                    processStepsDiv.textContent = `错误：${data.message}`;
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
            processStepsDiv.textContent = `错误：${error.message}`;
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
                    questionsList.replaceChildren();
                    questions.forEach((q, index) => {
                        const line = document.createElement('p');
                        line.textContent = `${index + 1}. ${q}`;
                        questionsList.appendChild(line);
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
                batchResultsContainer.textContent = `错误：${data.error}`;
                return;
            }
            
            batchResultsContainer.replaceChildren();
            data.results.forEach((result, index) => {
                const resultItem = document.createElement('div');
                resultItem.className = 'batch-result-item';
                
                if (result.error) {
                    const title = document.createElement('h4');
                    title.textContent = `问题 ${index + 1}: ${data.questions[index]}`;
                    const errorP = document.createElement('p');
                    errorP.style.color = 'red';
                    errorP.textContent = `错误：${result.error}`;
                    resultItem.appendChild(title);
                    resultItem.appendChild(errorP);
                } else {
                    const formattedAnswer = addCitationMarkers(result.answer, result.citations);
                    const title = document.createElement('h4');
                    title.textContent = `问题 ${index + 1}: ${data.questions[index]}`;
                    resultItem.appendChild(title);

                    const answerBlock = document.createElement('div');
                    answerBlock.className = 'batch-answer';
                    answerBlock.innerHTML = formattedAnswer;
                    resultItem.appendChild(answerBlock);

                    if (result.refs_gbt && result.refs_gbt.length > 0) {
                        const refsWrap = document.createElement('div');
                        refsWrap.className = 'batch-references';
                        const refsTitle = document.createElement('h5');
                        refsTitle.textContent = '参考文献：';
                        const refsOl = document.createElement('ol');
                        result.refs_gbt.forEach(ref => {
                            const li = document.createElement('li');
                            li.textContent = ref;
                            refsOl.appendChild(li);
                        });
                        refsWrap.appendChild(refsTitle);
                        refsWrap.appendChild(refsOl);
                        resultItem.appendChild(refsWrap);
                    }
                }
                
                batchResultsContainer.appendChild(resultItem);
            });
        })
        .catch(error => {
            batchResultsContainer.textContent = `请求失败：${error.message}`;
        });
    });
}

// 添加引用角标
function addCitationMarkers(answer, citations) {
    if (!citations || citations.length === 0) {
        return formatAnswerHtml(answer);
    }
    
    let result = formatAnswerHtml(answer);
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
        
        const queryDiv = document.createElement('div');
        queryDiv.className = 'history-query';
        queryDiv.textContent = record.query;

        const timeDiv = document.createElement('div');
        timeDiv.className = 'history-time';
        timeDiv.textContent = formattedTime;

        const answerDiv = document.createElement('div');
        answerDiv.className = 'history-answer';
        answerDiv.textContent = answerPreview;

        historyItem.appendChild(queryDiv);
        historyItem.appendChild(timeDiv);
        historyItem.appendChild(answerDiv);
        
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

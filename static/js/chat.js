/**
 * AI Data Analyst Agent - Phase 10: "Ask Your Dataset" Chat Interface Controller.
 * Manages conversational stream, deterministic tool calling execution, Plotly chart rendering,
 * session memory management, dynamic question suggestions, and markdown rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    initChatInterface();
});

let currentDatasetId = '';
let activeProvider = 'deterministic_fallback';
let isProcessing = false;

/**
 * Initialize the Ask Your Dataset Q&A workspace.
 */
async function initChatInterface() {
    const urlParams = new URLSearchParams(window.location.search);
    currentDatasetId = urlParams.get('dataset_id') || '';

    setupEventListeners();
    await loadAvailableDatasets();

    if (currentDatasetId) {
        await loadDatasetContext(currentDatasetId);
        await loadConversationHistory(currentDatasetId);
        await loadSuggestedQuestions(currentDatasetId);
    } else {
        showNoDatasetSelectedState();
    }
}

/**
 * Wire up UI event listeners.
 */
function setupEventListeners() {
    const sendBtn = document.getElementById('chat-send-btn');
    const input = document.getElementById('chat-input');
    const clearBtn = document.getElementById('clear-chat-btn');
    const exportBtn = document.getElementById('export-chat-btn');
    const datasetSelect = document.getElementById('dataset-switcher-select');

    if (sendBtn) {
        sendBtn.addEventListener('click', handleSendMessage);
    }

    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', handleClearConversation);
    }

    if (exportBtn) {
        exportBtn.addEventListener('click', handleExportTranscript);
    }

    if (datasetSelect) {
        datasetSelect.addEventListener('change', (e) => {
            const selected = e.target.value;
            if (selected && selected !== currentDatasetId) {
                window.location.href = `/chat?dataset_id=${encodeURIComponent(selected)}`;
            }
        });
    }
}

/**
 * Load all uploaded datasets into switcher dropdown.
 */
async function loadAvailableDatasets() {
    try {
        const res = await fetch('/api/datasets');
        if (!res.ok) return;
        const data = await res.json();
        const datasets = data.datasets || [];
        const select = document.getElementById('dataset-switcher-select');

        if (select) {
            select.innerHTML = '<option value="">-- Select Active Dataset --</option>';
            datasets.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.dataset_id;
                opt.textContent = `${ds.original_filename} (${ds.total_rows || 0} rows)`;
                if (ds.dataset_id === currentDatasetId) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });

            // If no dataset selected in URL, default to the most recent uploaded dataset
            if (!currentDatasetId && datasets.length > 0) {
                currentDatasetId = datasets[0].dataset_id;
                select.value = currentDatasetId;
                window.history.replaceState({}, '', `/chat?dataset_id=${encodeURIComponent(currentDatasetId)}`);
                await loadDatasetContext(currentDatasetId);
                await loadConversationHistory(currentDatasetId);
                await loadSuggestedQuestions(currentDatasetId);
            }
        }
    } catch (err) {
        console.warn('Failed to load datasets list:', err);
    }
}

/**
 * Load dataset metadata and column badges in sidebar.
 */
async function loadDatasetContext(datasetId) {
    const dsIdEl = document.getElementById('chat-dataset-id');
    const colsContainer = document.getElementById('sidebar-columns-list');
    const metaStatsEl = document.getElementById('sidebar-dataset-meta');
    const dashLink = document.getElementById('nav-dashboard');

    if (dsIdEl) dsIdEl.textContent = datasetId;
    if (dashLink) dashLink.href = `/dashboard?dataset_id=${encodeURIComponent(datasetId)}`;

    try {
        const res = await fetch(`/api/visualization/schema/${encodeURIComponent(datasetId)}`);
        if (!res.ok) return;
        const data = await res.json();
        const schema = data.schema || {};

        if (colsContainer) {
            colsContainer.innerHTML = '';
            const typeBadges = [
                { type: 'Numerical', key: 'numerical', icon: '🔢', color: '#818cf8' },
                { type: 'Categorical', key: 'categorical', icon: '🏷️', color: '#a855f7' },
                { type: 'Datetime', key: 'datetime', icon: '📅', color: '#38bdf8' },
            ];

            typeBadges.forEach(tb => {
                const cols = schema[tb.key] || [];
                if (cols.length > 0) {
                    const groupDiv = document.createElement('div');
                    groupDiv.className = 'col-schema-group';
                    groupDiv.innerHTML = `
                        <span class="col-type-label">${tb.icon} ${tb.type} (${cols.length})</span>
                        <div class="col-chips-wrap">
                            ${cols.map(c => `<button class="col-tag-chip" onclick="insertPromptColumn('${escapeHtml(c)}')">${escapeHtml(c)}</button>`).join('')}
                        </div>
                    `;
                    colsContainer.appendChild(groupDiv);
                }
            });
        }
    } catch (err) {
        console.warn('Failed to load dataset schema:', err);
    }
}

/**
 * Load dynamic question suggestions tailored to the active dataset's columns.
 */
async function loadSuggestedQuestions(datasetId) {
    const container = document.getElementById('suggested-prompts-container');
    if (!container) return;

    try {
        const res = await fetch(`/api/chat/suggested-questions/${encodeURIComponent(datasetId)}`);
        if (res.ok) {
            const data = await res.json();
            const suggestions = data.suggestions || [];
            if (suggestions.length > 0) {
                container.innerHTML = suggestions.map(q => `
                    <button class="prompt-chip" onclick="askPresetPrompt('${escapeHtml(q)}')">
                        <span class="chip-spark">✨</span>
                        <span>"${escapeHtml(q)}"</span>
                    </button>
                `).join('');
                return;
            }
        }
    } catch (err) {
        console.warn('Failed to load suggestions:', err);
    }

    // Default sample questions from requirements
    const defaultPrompts = [
        "Which category has the highest sales?",
        "What is the average profit?",
        "Which region performs best?",
        "What are the strongest correlations?",
        "Are there unusual values?",
        "How has sales changed over time?",
        "Which products have the highest revenue?",
        "What should I investigate further?",
    ];

    container.innerHTML = defaultPrompts.map(q => `
        <button class="prompt-chip" onclick="askPresetPrompt('${escapeHtml(q)}')">
            <span class="chip-spark">💡</span>
            <span>"${escapeHtml(q)}"</span>
        </button>
    `).join('');
}

/**
 * Load conversation turns from session history.
 */
async function loadConversationHistory(datasetId) {
    const stream = document.getElementById('messages-container');
    if (!stream) return;

    try {
        const res = await fetch(`/api/chat/history/${encodeURIComponent(datasetId)}`);
        if (!res.ok) return;
        const data = await res.json();
        const history = data.history || [];

        if (history.length === 0) {
            renderWelcomeHero();
            return;
        }

        stream.innerHTML = '';
        history.forEach(turn => {
            renderUserMessage(turn.user_query, turn.timestamp);
            renderAssistantMessage({
                tool_executed: turn.tool_executed,
                tool_parameters: turn.tool_params,
                tool_result: turn.tool_result,
                explanation: turn.assistant_explanation,
                visualization: turn.visualization,
                followups: turn.followups,
                turn_id: turn.id,
            });
        });

        scrollChatToBottom();
    } catch (err) {
        console.warn('Failed to load history:', err);
    }
}

/**
 * Handle sending a question from the input bar.
 */
async function handleSendMessage() {
    const input = document.getElementById('chat-input');
    if (!input || isProcessing) return;

    const query = input.value.trim();
    if (!query) return;

    if (!currentDatasetId) {
        alert('Please select or upload a dataset first.');
        return;
    }

    input.value = '';
    await processQuestion(query);
}

/**
 * Trigger question from quick prompt chip.
 */
window.askPresetPrompt = function(promptText) {
    if (isProcessing) return;
    const input = document.getElementById('chat-input');
    if (input) input.value = promptText;
    processQuestion(promptText);
};

/**
 * Insert column name into input.
 */
window.insertPromptColumn = function(colName) {
    const input = document.getElementById('chat-input');
    if (input) {
        input.value = input.value ? `${input.value} ${colName}` : `Tell me about ${colName}`;
        input.focus();
    }
};

/**
 * Process question through the backend Phase 10 engine.
 */
async function processQuestion(query) {
    isProcessing = true;
    setSendButtonLoading(true);

    // Remove welcome hero if present
    const hero = document.getElementById('chat-welcome-hero');
    if (hero) hero.remove();

    renderUserMessage(query, new Date().toISOString());
    const typingId = renderTypingIndicator();
    scrollChatToBottom();

    try {
        const res = await fetch('/api/chat/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: currentDatasetId,
                query: query,
            }),
        });

        removeTypingIndicator(typingId);

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            renderErrorMessage(errData.error || 'Server error occurred while executing data analysis.');
            return;
        }

        const data = await res.json();
        renderAssistantMessage(data);
    } catch (err) {
        removeTypingIndicator(typingId);
        renderErrorMessage(`Network connection error: ${err.message}`);
    } finally {
        isProcessing = false;
        setSendButtonLoading(false);
        scrollChatToBottom();
    }
}

/**
 * Render user message bubble.
 */
function renderUserMessage(text, timestamp) {
    const stream = document.getElementById('messages-container');
    if (!stream) return;

    const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user animate-fade-in';
    bubble.innerHTML = `
        <div class="message-avatar user-avatar">👤</div>
        <div class="message-content user-content">
            <p class="user-text">${escapeHtml(text)}</p>
            ${timeStr ? `<span class="message-timestamp">${timeStr}</span>` : ''}
        </div>
    `;
    stream.appendChild(bubble);
}

/**
 * Render assistant response card with tool badge, grounded explanation,
 * embedded Plotly chart, raw JSON toggle, and follow-up chips.
 */
function renderAssistantMessage(data) {
    const stream = document.getElementById('messages-container');
    if (!stream) return;

    const turnId = data.turn_id || `turn_${Date.now()}`;
    const chartId = `chart_${turnId}`;
    const jsonDrawerId = `json_${turnId}`;
    const planDrawerId = `plan_${turnId}`;
    const toolName = data.tool_executed || 'dataset_summary';
    const toolParamsStr = JSON.stringify(data.tool_parameters || {}, null, 1).replace(/\n\s*/g, ' ');

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble assistant animate-fade-in';

    // Format markdown explanation to HTML
    const formattedHtml = formatMarkdown(data.explanation || 'No response synthesized.');

    // Analysis Plan HTML (Phase 11)
    const planSteps = data.analysis_plan || [];
    const requiresViz = data.requires_visualization;
    const chartType = data.recommended_chart_type || (data.visualization ? data.visualization.chart_type : 'none');
    const vizReason = data.visualization_reasoning || '';

    const planHtml = planSteps.length > 0 ? `
        <div class="analysis-plan-card">
            <div class="analysis-plan-header" onclick="togglePlanDrawer('${planDrawerId}')">
                <div class="plan-header-title">
                    <span class="plan-icon">📋</span>
                    <span class="plan-title-text">Execution Plan (${planSteps.length} Steps)</span>
                </div>
                <div class="plan-header-meta">
                    <span class="plan-viz-badge ${requiresViz ? 'viz-needed' : 'viz-not-needed'}">
                        ${requiresViz ? `📊 ${escapeHtml(chartType.toUpperCase())} CHART` : 'ℹ️ SCALAR METRIC'}
                    </span>
                    <span class="plan-arrow" id="arrow-${planDrawerId}">▼</span>
                </div>
            </div>
            <div class="analysis-plan-body" id="${planDrawerId}">
                <ol class="plan-steps-list">
                    ${planSteps.map(step => `
                        <li class="plan-step-item">
                            <span class="step-check">✓</span>
                            <span class="step-text">${escapeHtml(step.replace(/^\d+\.\s*/, ''))}</span>
                        </li>
                    `).join('')}
                </ol>
                ${vizReason ? `<div class="plan-viz-reasoning"><strong>Visualization Decision:</strong> ${escapeHtml(vizReason)}</div>` : ''}
            </div>
        </div>
    ` : '';

    // Follow-ups HTML
    const followups = data.followups || [];
    const followupsHtml = followups.length > 0 ? `
        <div class="followup-suggestions-block">
            <span class="followup-label">💡 Suggested Follow-up Questions:</span>
            <div class="followup-chips-wrap">
                ${followups.map(f => `<button class="followup-chip" onclick="askPresetPrompt('${escapeHtml(f)}')">${escapeHtml(f)}</button>`).join('')}
            </div>
        </div>
    ` : '';

    // Visualization Container HTML
    const vizHtml = data.visualization ? `
        <div class="chat-viz-wrapper" id="viz-wrap-${turnId}">
            <div class="viz-header-bar">
                <span class="viz-badge">📊 Interactive Chart</span>
                <span class="viz-type-tag">${escapeHtml(data.visualization.chart_type || 'Plotly')}</span>
            </div>
            <div class="chat-plotly-container" id="${chartId}"></div>
        </div>
    ` : '';

    // Raw JSON Tool Inspector HTML
    const jsonInspectorHtml = `
        <div class="tool-json-drawer">
            <button class="json-toggle-btn" onclick="toggleJsonDrawer('${jsonDrawerId}')">
                <span>🔍 Inspect Python Tool Calculation (JSON)</span>
                <span class="drawer-arrow" id="arrow-${jsonDrawerId}">▼</span>
            </button>
            <div class="json-drawer-content" id="${jsonDrawerId}" style="display: none;">
                <pre class="json-code-block"><code>${escapeHtml(JSON.stringify(data.tool_result || {}, null, 2))}</code></pre>
            </div>
        </div>
    `;

    bubble.innerHTML = `
        <div class="message-avatar assistant-avatar">🤖</div>
        <div class="message-content assistant-content">
            <!-- Tool Badge Header -->
            <div class="tool-execution-pill">
                <span class="pill-spark">⚡</span>
                <span class="tool-name-highlight">Tool: <code>${escapeHtml(toolName)}</code></span>
                ${data.tool_parameters && Object.keys(data.tool_parameters).length > 0 ? `<span class="tool-params-preview">${escapeHtml(toolParamsStr)}</span>` : ''}
                <span class="tool-grounded-tag">✓ Controlled Python Execution</span>
            </div>

            <!-- Execution Plan Stepper Card -->
            ${planHtml}

            <!-- Grounded Explanation Text -->
            <div class="assistant-markdown-body">
                ${formattedHtml}
            </div>

            <!-- Embedded Interactive Chart (if generated) -->
            ${vizHtml}

            <!-- JSON Output Drawer -->
            ${jsonInspectorHtml}

            <!-- Follow-up chips -->
            ${followupsHtml}
        </div>
    `;

    stream.appendChild(bubble);

    // Render Plotly chart if spec provided
    if (data.visualization && window.Plotly) {
        try {
            const chartDiv = document.getElementById(chartId);
            if (chartDiv) {
                const spec = data.visualization;
                Plotly.newPlot(chartDiv, spec.data || [], spec.layout || {}, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                });
            }
        } catch (e) {
            console.warn('Plotly rendering error:', e);
        }
    }
}

/**
 * Toggle execution plan drawer.
 */
window.togglePlanDrawer = function(drawerId) {
    const el = document.getElementById(drawerId);
    const arrow = document.getElementById(`arrow-${drawerId}`);
    if (el) {
        const isHidden = el.style.display === 'none';
        el.style.display = isHidden ? 'block' : 'none';
        if (arrow) arrow.textContent = isHidden ? '▼' : '▲';
    }
};

/**
 * Toggle raw JSON inspection drawer.
 */
window.toggleJsonDrawer = function(drawerId) {
    const el = document.getElementById(drawerId);
    const arrow = document.getElementById(`arrow-${drawerId}`);
    if (el) {
        const isHidden = el.style.display === 'none';
        el.style.display = isHidden ? 'block' : 'none';
        if (arrow) arrow.textContent = isHidden ? '▲' : '▼';
    }
};

/**
 * Typing indicator.
 */
function renderTypingIndicator() {
    const stream = document.getElementById('messages-container');
    if (!stream) return null;

    const id = `typing_${Date.now()}`;
    const indicator = document.createElement('div');
    indicator.className = 'message-bubble assistant typing-bubble';
    indicator.id = id;
    indicator.innerHTML = `
        <div class="message-avatar assistant-avatar">🤖</div>
        <div class="message-content assistant-content typing-content">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
            <span class="typing-label">Executing Python analytical engine & grounding response...</span>
        </div>
    `;
    stream.appendChild(indicator);
    return id;
}

function removeTypingIndicator(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
}

/**
 * Render error message.
 */
function renderErrorMessage(msg) {
    const stream = document.getElementById('messages-container');
    if (!stream) return;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble assistant error-bubble';
    bubble.innerHTML = `
        <div class="message-avatar assistant-avatar">⚠️</div>
        <div class="message-content error-content">
            <h4>Execution Notice</h4>
            <p>${escapeHtml(msg)}</p>
        </div>
    `;
    stream.appendChild(bubble);
}

/**
 * Welcome hero state when history is empty.
 */
function renderWelcomeHero() {
    const stream = document.getElementById('messages-container');
    if (!stream) return;

    stream.innerHTML = `
        <div class="chat-welcome-hero" id="chat-welcome-hero">
            <div class="welcome-icon-large">⚡</div>
            <h2>Ask Your Dataset</h2>
            <p class="welcome-subtitle">
                Ask high-level business questions, discover correlations, detect anomalies, and explore statistical breakdowns in natural language.
            </p>
            <div class="zero-hallucination-badge">
                <span>🔒 100% Deterministic Python Execution — Zero Statistical Hallucinations</span>
            </div>

            <div class="quick-starter-grid">
                <div class="starter-card" onclick="askPresetPrompt('Which category has the highest sales?')">
                    <span class="starter-icon">🏆</span>
                    <h4>Top Performer Breakdown</h4>
                    <p>"Which category has the highest sales?"</p>
                </div>
                <div class="starter-card" onclick="askPresetPrompt('What is the average sales amount?')">
                    <span class="starter-icon">📊</span>
                    <h4>Metric Aggregation</h4>
                    <p>"What is the average sales amount?"</p>
                </div>
                <div class="starter-card" onclick="askPresetPrompt('What are the strongest correlations?')">
                    <span class="starter-icon">🔗</span>
                    <h4>Correlation Analysis</h4>
                    <p>"What are the strongest correlations?"</p>
                </div>
                <div class="starter-card" onclick="askPresetPrompt('Are there unusual values?')">
                    <span class="starter-icon">⚠️</span>
                    <h4>Outlier & Anomaly Scan</h4>
                    <p>"Are there unusual values?"</p>
                </div>
            </div>
        </div>
    `;
}

function showNoDatasetSelectedState() {
    const stream = document.getElementById('messages-container');
    if (stream) {
        stream.innerHTML = `
            <div class="chat-welcome-hero">
                <div class="welcome-icon-large">📂</div>
                <h2>No Active Dataset Selected</h2>
                <p>Please upload a CSV dataset on the <a href="/" class="link-highlight">Home Page</a> or select a dataset from the dropdown above.</p>
            </div>
        `;
    }
}

/**
 * Clear conversation session.
 */
async function handleClearConversation() {
    if (!currentDatasetId) return;
    if (!confirm('Are you sure you want to clear the conversation history for this dataset?')) return;

    try {
        const res = await fetch(`/api/chat/clear/${encodeURIComponent(currentDatasetId)}`, { method: 'POST' });
        if (res.ok) {
            renderWelcomeHero();
        }
    } catch (err) {
        console.warn('Failed to clear history:', err);
    }
}

/**
 * Export conversation transcript.
 */
async function handleExportTranscript() {
    if (!currentDatasetId) return;

    try {
        const res = await fetch(`/api/chat/history/${encodeURIComponent(currentDatasetId)}`);
        if (!res.ok) return;
        const data = await res.json();
        const history = data.history || [];

        if (history.length === 0) {
            alert('No conversation history to export.');
            return;
        }

        let markdown = `# AI Data Analyst Agent - Chat Transcript\n`;
        markdown += `**Dataset ID**: \`${currentDatasetId}\`\n`;
        markdown += `**Exported At**: ${new Date().toISOString()}\n\n---\n\n`;

        history.forEach((turn, idx) => {
            markdown += `### Turn ${idx + 1}: User Question\n`;
            markdown += `> ${turn.user_query}\n\n`;
            markdown += `**Tool Executed**: \`${turn.tool_executed}\`\n\n`;
            markdown += `**Assistant Explanation**:\n${turn.assistant_explanation}\n\n---\n\n`;
        });

        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_transcript_${currentDatasetId}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        console.warn('Export failed:', err);
    }
}

/**
 * Scroll stream to bottom.
 */
function scrollChatToBottom() {
    const stream = document.getElementById('messages-container');
    if (stream) {
        stream.scrollTop = stream.scrollHeight;
    }
}

function setSendButtonLoading(loading) {
    const btn = document.getElementById('chat-send-btn');
    if (!btn) return;
    btn.disabled = loading;
    btn.innerHTML = loading ? '<span class="spinner-small"></span>' : '<span>Send</span>';
}

/**
 * Lightweight client-side Markdown formatter for Q&A responses.
 */
function formatMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');

    // Code blocks & inline code
    html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');

    // Blockquotes
    html = html.replace(/^\&gt; (.*$)/gim, '<blockquote class="grounding-quote">$1</blockquote>');

    // Tables
    const tableRegex = /\|(.+)\|\n\|(?:\s*:?-+:?\s*\|)+\n((?:\|.+\|\n?)+)/g;
    html = html.replace(tableRegex, (match, headerRow, bodyRows) => {
        const headers = headerRow.split('|').filter(h => h.trim()).map(h => `<th>${h.trim()}</th>`).join('');
        const rows = bodyRows.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<div class="table-responsive-wrapper"><table class="chat-md-table"><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
    });

    // Unordered lists
    html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\s*)+)/gim, '<ul class="chat-md-list">$1</ul>');

    // Paragraph line breaks
    html = html.replace(/\n\n+/gim, '<br/><br/>');

    return html;
}

function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

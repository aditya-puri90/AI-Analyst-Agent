/**
 * Dashboard Client-Side Controller for AI Data Analyst Agent.
 * Manages dataset selection, profile visualization, interactive schema inspection, and paginated table exploration.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let activeDatasetId = window.INITIAL_DATASET_ID || '';
    let currentPage = 1;
    const pageSize = 20;
    let currentPreviewRows = [];
    let currentPreviewCols = [];

    // DOM Elements
    const datasetSelector = document.getElementById('dataset-selector');
    const loadingState = document.getElementById('dashboard-loading');
    const errorState = document.getElementById('dashboard-error');
    const contentState = document.getElementById('dashboard-content');
    const activeDatasetName = document.getElementById('active-dataset-name');
    const activeDatasetIdEl = document.getElementById('active-dataset-id');
    const activeDatasetSize = document.getElementById('active-dataset-size');
    const openChatBtn = document.getElementById('open-chat-btn');

    // KPI Elements
    const kpiRows = document.getElementById('kpi-rows');
    const kpiCols = document.getElementById('kpi-cols');
    const kpiMemory = document.getElementById('kpi-memory');
    const kpiMissing = document.getElementById('kpi-missing');
    const kpiMissingHint = document.getElementById('kpi-missing-hint');
    const kpiDuplicates = document.getElementById('kpi-duplicates');
    const kpiDuplicatesHint = document.getElementById('kpi-duplicates-hint');
    const typeBadgesGrid = document.getElementById('type-badges-grid');
    const columnCardsGrid = document.getElementById('column-cards-grid');

    // Explorer Elements
    const tableHead = document.getElementById('data-table-head');
    const tableBody = document.getElementById('data-table-body');
    const paginationInfo = document.getElementById('pagination-info');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const tableSearch = document.getElementById('table-search');

    // Tab Navigation
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            tabButtons.forEach((b) => b.classList.remove('active'));
            tabPanes.forEach((p) => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // Helper: Format bytes
    function formatBytes(bytes, decimals = 2) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Helper: Escape HTML
    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Load available datasets for the dropdown
    async function loadDatasetsList() {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();
            if (res.ok && data.success && data.datasets) {
                if (datasetSelector) {
                    datasetSelector.innerHTML = '<option value="">-- Switch Dataset --</option>';
                    data.datasets.forEach((ds) => {
                        const opt = document.createElement('option');
                        opt.value = ds.id;
                        opt.textContent = `${ds.original_name} (${formatBytes(ds.size_bytes)})`;
                        if (ds.id === activeDatasetId) {
                            opt.selected = true;
                        }
                        datasetSelector.appendChild(opt);
                    });
                }

                // If no activeDatasetId, default to latest dataset
                if (!activeDatasetId && data.datasets.length > 0) {
                    activeDatasetId = data.datasets[0].id;
                    if (datasetSelector) datasetSelector.value = activeDatasetId;
                }
            }
        } catch (e) {
            console.error('Failed to load datasets list:', e);
        }
    }

    // Dataset Selector change
    if (datasetSelector) {
        datasetSelector.addEventListener('change', (e) => {
            const selectedId = e.target.value;
            if (selectedId) {
                activeDatasetId = selectedId;
                const newUrl = `${window.location.pathname}?dataset_id=${encodeURIComponent(selectedId)}`;
                window.history.pushState({ path: newUrl }, '', newUrl);
                loadDatasetDashboard(selectedId);
            }
        });
    }

    // Main Dashboard Loader
    async function loadDatasetDashboard(datasetId) {
        if (!datasetId) {
            if (loadingState) loadingState.classList.add('hidden');
            if (errorState) errorState.classList.remove('hidden');
            if (contentState) contentState.classList.add('hidden');
            return;
        }

        if (loadingState) loadingState.classList.remove('hidden');
        if (errorState) errorState.classList.add('hidden');
        if (contentState) contentState.classList.add('hidden');

        // Update links
        if (openChatBtn) openChatBtn.href = `/chat?dataset_id=${encodeURIComponent(datasetId)}`;

        try {
            const res = await fetch(`/api/profile/${encodeURIComponent(datasetId)}`);
            const data = await res.json();

            if (!res.ok || !data.success) {
                showDashboardError(data.error || 'Could not profile the selected dataset.');
                return;
            }

            const profile = data.profile;
            renderProfileOverview(datasetId, profile);
            renderSchemaInspector(profile.columns, profile.overview.type_counts);

            // Load table preview
            currentPage = 1;
            await loadTablePreview(datasetId, currentPage);

            if (loadingState) loadingState.classList.add('hidden');
            if (contentState) contentState.classList.remove('hidden');

        } catch (err) {
            showDashboardError(`Error loading dataset: ${err.message}`);
        }
    }

    function showDashboardError(msg) {
        if (loadingState) loadingState.classList.add('hidden');
        if (contentState) contentState.classList.add('hidden');
        if (errorState) {
            const errMsg = document.getElementById('error-message');
            if (errMsg) errMsg.textContent = msg;
            errorState.classList.remove('hidden');
        }
    }

    function renderProfileOverview(datasetId, profile) {
        const overview = profile.overview;

        if (activeDatasetName) activeDatasetName.textContent = datasetId.split('_').slice(2).join('_') || datasetId;
        if (activeDatasetIdEl) activeDatasetIdEl.textContent = datasetId;
        if (activeDatasetSize) activeDatasetSize.textContent = `${overview.memory_usage_mb} MB in RAM`;

        if (kpiRows) kpiRows.textContent = overview.total_rows.toLocaleString();
        if (kpiCols) kpiCols.textContent = overview.total_columns.toLocaleString();
        if (kpiMemory) kpiMemory.textContent = `${overview.memory_usage_mb} MB`;
        
        if (kpiMissing) kpiMissing.textContent = overview.total_missing_cells.toLocaleString();
        if (kpiMissingHint) kpiMissingHint.textContent = `${overview.missing_cells_percentage}% total missingness`;

        if (kpiDuplicates) kpiDuplicates.textContent = overview.duplicate_rows.toLocaleString();
        if (kpiDuplicatesHint) kpiDuplicatesHint.textContent = `${overview.duplicate_rows_percentage}% redundant rows`;
    }

    function renderSchemaInspector(columns, typeCounts) {
        // 1. Inferred Type Badges
        if (typeBadgesGrid) {
            typeBadgesGrid.innerHTML = '';
            const typeIcons = {
                numerical: '🔢 Numerical',
                categorical: '🏷️ Categorical',
                datetime: '📅 Datetime',
                boolean: '⚖️ Boolean',
                id_or_text: '🔤 ID / Text',
                unknown: '❓ Unknown'
            };

            for (const [type, count] of Object.entries(typeCounts || {})) {
                const badge = document.createElement('div');
                badge.className = `type-badge type-${type}`;
                badge.innerHTML = `
                    <span>${typeIcons[type] || type}</span>
                    <span class="type-badge-count">${count}</span>
                `;
                typeBadgesGrid.appendChild(badge);
            }
        }

        // 2. Column Cards Grid
        if (columnCardsGrid) {
            columnCardsGrid.innerHTML = '';
            columns.forEach((col) => {
                const card = document.createElement('div');
                card.className = 'col-card';

                const sampleChipsHtml = (col.sample_values || [])
                    .map((val) => `<span class="sample-chip" title="${escapeHtml(val)}">${escapeHtml(val !== null ? val : 'NULL')}</span>`)
                    .join('');

                card.innerHTML = `
                    <div class="col-card-top">
                        <span class="col-name">${escapeHtml(col.name)}</span>
                        <span class="type-pill type-badge type-${col.inferred_type}">${escapeHtml(col.inferred_type)}</span>
                    </div>
                    <div class="col-stats-row">
                        <span>Non-Null: <strong>${col.non_null_count.toLocaleString()}</strong> (${100 - col.null_percentage}%)</span>
                        <span>Nulls: <strong>${col.null_count.toLocaleString()}</strong> (${col.null_percentage}%)</span>
                    </div>
                    <div class="col-stats-row">
                        <span>Distinct Values: <strong>${col.unique_count.toLocaleString()}</strong></span>
                        <span>Dtype: <code>${escapeHtml(col.pandas_dtype)}</code></span>
                    </div>
                    <div class="col-samples-label">Sample Data:</div>
                    <div class="col-samples-chips">
                        ${sampleChipsHtml || '<span class="null-badge">No samples</span>'}
                    </div>
                `;
                columnCardsGrid.appendChild(card);
            });
        }
    }

    // Load Paginated Table Preview
    async function loadTablePreview(datasetId, page = 1) {
        try {
            const res = await fetch(`/api/preview/${encodeURIComponent(datasetId)}?page=${page}&page_size=${pageSize}`);
            const data = await res.json();

            if (!res.ok || !data.success) return;

            currentPreviewCols = data.columns || [];
            currentPreviewRows = data.rows || [];
            const pagination = data.pagination;

            renderTable(currentPreviewCols, currentPreviewRows);

            if (paginationInfo) {
                paginationInfo.textContent = `Page ${pagination.current_page} of ${pagination.total_pages} (${pagination.total_rows.toLocaleString()} rows)`;
            }

            if (prevPageBtn) prevPageBtn.disabled = !pagination.has_prev;
            if (nextPageBtn) nextPageBtn.disabled = !pagination.has_next;

        } catch (e) {
            console.error('Failed to load table preview:', e);
        }
    }

    function renderTable(columns, rows) {
        if (!tableHead || !tableBody) return;

        // Render Head
        tableHead.innerHTML = `
            <tr>
                <th style="width: 40px;">#</th>
                ${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}
            </tr>
        `;

        // Render Body
        if (rows.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="${columns.length + 1}" style="text-align: center; color: var(--text-muted);">No matching records found.</td></tr>`;
            return;
        }

        const startIdx = (currentPage - 1) * pageSize;
        tableBody.innerHTML = rows.map((row, idx) => {
            const rowNumber = startIdx + idx + 1;
            const cells = columns.map((col) => {
                const val = row[col];
                if (val === null || val === undefined) {
                    return `<td><span class="null-badge">NULL</span></td>`;
                }
                return `<td>${escapeHtml(val)}</td>`;
            }).join('');

            return `<tr><td style="color: var(--text-muted); font-family: var(--font-mono);">${rowNumber}</td>${cells}</tr>`;
        }).join('');
    }

    // Table Search Filter (Filters rows currently in the current page preview)
    if (tableSearch) {
        tableSearch.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderTable(currentPreviewCols, currentPreviewRows);
                return;
            }

            const filtered = currentPreviewRows.filter((row) => {
                return Object.values(row).some((val) => {
                    return val !== null && val !== undefined && String(val).toLowerCase().includes(query);
                });
            });
            renderTable(currentPreviewCols, filtered);
        });
    }

    // Pagination Click Handlers
    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', async () => {
            if (currentPage > 1) {
                currentPage--;
                await loadTablePreview(activeDatasetId, currentPage);
            }
        });
    }

    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', async () => {
            currentPage++;
            await loadTablePreview(activeDatasetId, currentPage);
        });
    }

    // Initial Execution
    (async () => {
        await loadDatasetsList();
        await loadDatasetDashboard(activeDatasetId);
    })();
});

/**
 * Dashboard Client-Side Controller for AI Data Analyst Agent.
 * Phase 3: Automatic Dataset Profiling, Schema Inference, Data Quality Scoring, and Exploration.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let activeDatasetId = window.INITIAL_DATASET_ID || '';
    let currentProfile = null;
    let allColumnsProfile = [];
    let activeTypeFilter = 'all';
    let columnSearchQuery = '';
    
    let currentPage = 1;
    const pageSize = 20;
    let currentPreviewRows = [];
    let currentPreviewCols = [];

    // DOM Elements - Global
    const datasetSelector = document.getElementById('dataset-selector');
    const loadingState = document.getElementById('dashboard-loading');
    const errorState = document.getElementById('dashboard-error');
    const contentState = document.getElementById('dashboard-content');
    const activeDatasetName = document.getElementById('active-dataset-name');
    const activeDatasetIdEl = document.getElementById('active-dataset-id');
    const activeDatasetSize = document.getElementById('active-dataset-size');
    const openChatBtn = document.getElementById('open-chat-btn');

    // DOM Elements - Overview KPIs
    const kpiRows = document.getElementById('kpi-rows');
    const kpiCols = document.getElementById('kpi-cols');
    const kpiMemory = document.getElementById('kpi-memory');
    const kpiMemoryBytes = document.getElementById('kpi-memory-bytes');
    const kpiMissing = document.getElementById('kpi-missing');
    const kpiMissingHint = document.getElementById('kpi-missing-hint');
    const kpiDuplicates = document.getElementById('kpi-duplicates');
    const kpiDuplicatesHint = document.getElementById('kpi-duplicates-hint');
    const kpiHealthGrade = document.getElementById('kpi-health-grade');
    const kpiHealthScore = document.getElementById('kpi-health-score');

    // DOM Elements - Quality Summary
    const qualityStatusBadge = document.getElementById('quality-status-badge');
    const qualityStatusText = document.getElementById('quality-status-text');
    const qualityScoreDisplay = document.getElementById('quality-score-display');
    const qualityChecksCount = document.getElementById('quality-checks-count');
    const completenessScoreVal = document.getElementById('completeness-score-val');
    const completenessBarFill = document.getElementById('completeness-bar-fill');
    const uniquenessScoreVal = document.getElementById('uniqueness-score-val');
    const uniquenessBarFill = document.getElementById('uniqueness-bar-fill');
    const qualityInsightsList = document.getElementById('quality-insights-list');

    // DOM Elements - Column Profiling Table
    const countAll = document.getElementById('count-all');
    const countNumerical = document.getElementById('count-numerical');
    const countCategorical = document.getElementById('count-categorical');
    const countDatetime = document.getElementById('count-datetime');
    const countBoolean = document.getElementById('count-boolean');
    const countOther = document.getElementById('count-other');
    const typeFilterBtns = document.querySelectorAll('.type-filter-btn');
    const columnSearch = document.getElementById('column-search');
    const columnTableMeta = document.getElementById('column-table-meta');
    const columnsProfileTbody = document.getElementById('columns-profile-tbody');

    // DOM Elements - Modal Inspector
    const modalBackdrop = document.getElementById('column-modal-backdrop');
    const modalColName = document.getElementById('modal-col-name');
    const modalColType = document.getElementById('modal-col-type');
    const modalColDtype = document.getElementById('modal-col-dtype');
    const modalColBody = document.getElementById('modal-col-body');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // DOM Elements - Preview Explorer
    const tableHead = document.getElementById('data-table-head');
    const tableBody = document.getElementById('data-table-body');
    const paginationInfo = document.getElementById('pagination-info');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const tableSearch = document.getElementById('table-search');

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

            currentProfile = data.profile;
            allColumnsProfile = currentProfile.columns || [];

            renderOverview(datasetId, currentProfile.overview, currentProfile.quality);
            renderQualitySummary(currentProfile.quality);
            updateTypeCounts(currentProfile.type_counts, allColumnsProfile.length);
            renderColumnProfilingTable();

            // Load table preview
            currentPage = 1;
            await loadTablePreview(datasetId, currentPage);

            if (loadingState) loadingState.classList.add('hidden');
            if (contentState) contentState.classList.remove('hidden');

        } catch (err) {
            showDashboardError(`Error profiling dataset: ${err.message}`);
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

    // 1. Render Dataset Overview Cards
    function renderOverview(datasetId, overview, quality) {
        const rawName = datasetId.split('_').slice(2).join('_') || datasetId;
        if (activeDatasetName) activeDatasetName.textContent = rawName;
        if (activeDatasetIdEl) activeDatasetIdEl.textContent = datasetId;
        if (activeDatasetSize) activeDatasetSize.textContent = `${overview.memory_usage_formatted} RAM`;

        if (kpiRows) kpiRows.textContent = overview.total_rows.toLocaleString();
        if (kpiCols) kpiCols.textContent = overview.total_columns.toLocaleString();
        if (kpiMemory) kpiMemory.textContent = overview.memory_usage_formatted;
        if (kpiMemoryBytes) kpiMemoryBytes.textContent = `${overview.memory_usage_bytes.toLocaleString()} bytes in memory`;
        
        if (kpiMissing) kpiMissing.textContent = overview.total_missing_cells.toLocaleString();
        if (kpiMissingHint) kpiMissingHint.textContent = `${overview.missing_cells_percentage}% total missingness (${overview.rows_with_missing} rows affected)`;

        if (kpiDuplicates) kpiDuplicates.textContent = overview.duplicate_rows.toLocaleString();
        if (kpiDuplicatesHint) kpiDuplicatesHint.textContent = `${overview.duplicate_rows_percentage}% duplicate row rate`;

        if (kpiHealthGrade) kpiHealthGrade.textContent = quality ? quality.health_grade : '--';
        if (kpiHealthScore) kpiHealthScore.textContent = quality ? `Score: ${quality.health_score}/100` : 'Score: --';
    }

    // 2. Render Data Quality Summary
    function renderQualitySummary(quality) {
        if (!quality) return;

        if (qualityScoreDisplay) qualityScoreDisplay.textContent = quality.health_score;
        if (qualityChecksCount) qualityChecksCount.textContent = `Passed ${quality.passed_checks} of ${quality.total_checks} health checks`;

        if (qualityStatusText) qualityStatusText.textContent = quality.quality_status;
        if (qualityStatusBadge) {
            qualityStatusBadge.className = 'quality-status-badge';
            if (quality.health_score >= 85) {
                qualityStatusBadge.classList.add('status-success');
            } else if (quality.health_score >= 60) {
                qualityStatusBadge.classList.add('status-warning');
            } else {
                qualityStatusBadge.classList.add('status-danger');
            }
        }

        if (completenessScoreVal) completenessScoreVal.textContent = `${quality.completeness_score}%`;
        if (completenessBarFill) {
            completenessBarFill.style.width = `${quality.completeness_score}%`;
            completenessBarFill.className = 'progress-bar-fill';
            if (quality.completeness_score >= 90) completenessBarFill.classList.add('fill-emerald');
            else if (quality.completeness_score >= 70) completenessBarFill.classList.add('fill-amber');
            else completenessBarFill.classList.add('fill-rose');
        }

        if (uniquenessScoreVal) uniquenessScoreVal.textContent = `${quality.uniqueness_score}%`;
        if (uniquenessBarFill) {
            uniquenessBarFill.style.width = `${quality.uniqueness_score}%`;
            uniquenessBarFill.className = 'progress-bar-fill';
            if (quality.uniqueness_score >= 95) uniquenessBarFill.classList.add('fill-indigo');
            else if (quality.uniqueness_score >= 80) uniquenessBarFill.classList.add('fill-amber');
            else uniquenessBarFill.classList.add('fill-rose');
        }

        if (qualityInsightsList) {
            qualityInsightsList.innerHTML = '';
            (quality.warnings || []).forEach((w) => {
                const li = document.createElement('li');
                li.textContent = w;
                qualityInsightsList.appendChild(li);
            });
        }
    }

    // Update Filter Tab Counts
    function updateTypeCounts(typeCounts, totalCols) {
        if (countAll) countAll.textContent = totalCols;
        if (countNumerical) countNumerical.textContent = typeCounts['Numerical'] || 0;
        if (countCategorical) countCategorical.textContent = typeCounts['Categorical'] || 0;
        if (countDatetime) countDatetime.textContent = typeCounts['Datetime'] || 0;
        if (countBoolean) countBoolean.textContent = typeCounts['Boolean'] || 0;
        if (countOther) countOther.textContent = typeCounts['Other'] || 0;
    }

    // Filter Buttons Handlers
    typeFilterBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            typeFilterBtns.forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            activeTypeFilter = btn.getAttribute('data-type');
            renderColumnProfilingTable();
        });
    });

    // Column Search Input
    if (columnSearch) {
        columnSearch.addEventListener('input', (e) => {
            columnSearchQuery = e.target.value.toLowerCase().trim();
            renderColumnProfilingTable();
        });
    }

    // 3. Render Column Profiling Table
    function renderColumnProfilingTable() {
        if (!columnsProfileTbody) return;

        let filtered = allColumnsProfile;

        // Type filter
        if (activeTypeFilter !== 'all') {
            filtered = filtered.filter((c) => (c.classified_type || c.inferred_type) === activeTypeFilter);
        }

        // Search query filter
        if (columnSearchQuery) {
            filtered = filtered.filter((c) => {
                const nameMatch = c.name.toLowerCase().includes(columnSearchQuery);
                const dtypeMatch = c.pandas_dtype.toLowerCase().includes(columnSearchQuery);
                const typeMatch = (c.classified_type || '').toLowerCase().includes(columnSearchQuery);
                return nameMatch || dtypeMatch || typeMatch;
            });
        }

        if (columnTableMeta) {
            columnTableMeta.textContent = `Showing ${filtered.length} of ${allColumnsProfile.length} columns`;
        }

        if (filtered.length === 0) {
            columnsProfileTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 30px;">No columns match the selected filter.</td></tr>`;
            return;
        }

        columnsProfileTbody.innerHTML = filtered.map((col, idx) => {
            const colType = col.classified_type || col.inferred_type || 'Other';
            
            // Missing bar color
            let missingBarClass = 'fill-emerald';
            if (col.missing_percentage > 15) missingBarClass = 'fill-rose';
            else if (col.missing_percentage > 0) missingBarClass = 'fill-amber';

            // Stats summary content based on column type
            let statsSummaryHtml = '';

            if (colType === 'Numerical' && col.numerical_stats) {
                const ns = col.numerical_stats;
                statsSummaryHtml = `
                    <div class="stat-row">
                        <span class="stat-pill" title="Minimum and Maximum range">Range: <strong>[${ns.min}, ${ns.max}]</strong></span>
                        <span class="stat-pill" title="Mean ± Standard Deviation">Mean: <strong>${ns.mean}</strong> ± <strong>${ns.std}</strong></span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-pill" title="Median (50th percentile)">Median: <strong>${ns.median}</strong></span>
                        <span class="stat-pill" title="Interquartile Range (Q3 - Q1)">IQR: <strong>${ns.iqr}</strong></span>
                        <span class="stat-pill" title="Fisher-Pearson Skewness">Skew: <strong>${ns.skewness}</strong></span>
                    </div>
                `;
            } else if (colType === 'Categorical' && col.categorical_stats) {
                const cs = col.categorical_stats;
                statsSummaryHtml = `
                    <div class="stat-row">
                        <span class="stat-pill" title="Number of distinct categories">Categories: <strong>${cs.num_categories}</strong></span>
                        <span class="stat-pill" title="Most frequent category (Mode)">Top: <strong>${escapeHtml(cs.most_frequent_category)}</strong></span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-pill" title="Top category frequency">Freq: <strong>${cs.frequency_most_frequent.toLocaleString()}</strong> (${cs.frequency_percentage}%)</span>
                    </div>
                `;
            } else if (colType === 'Datetime' && col.datetime_stats) {
                const ds = col.datetime_stats;
                statsSummaryHtml = `
                    <div class="stat-row">
                        <span class="stat-pill" title="Earliest Date">Min: <strong>${ds.min_date ? ds.min_date.split('T')[0] : '--'}</strong></span>
                        <span class="stat-pill" title="Latest Date">Max: <strong>${ds.max_date ? ds.max_date.split('T')[0] : '--'}</strong></span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-pill" title="Date span">Span: <strong>${ds.date_range}</strong></span>
                    </div>
                `;
            } else if (colType === 'Boolean' && col.boolean_stats) {
                const bs = col.boolean_stats;
                statsSummaryHtml = `
                    <div class="stat-row">
                        <span class="stat-pill" style="color: var(--accent-emerald);">True: <strong>${bs.true_percentage}%</strong> (${bs.true_count})</span>
                        <span class="stat-pill" style="color: var(--accent-rose);">False: <strong>${bs.false_percentage}%</strong> (${bs.false_count})</span>
                    </div>
                `;
            } else if (col.other_stats) {
                const os = col.other_stats;
                statsSummaryHtml = `
                    <div class="stat-row">
                        <span class="stat-pill">Avg Length: <strong>${os.avg_length} chars</strong></span>
                        <span class="stat-pill">Span: <strong>[${os.min_length}, ${os.max_length}]</strong></span>
                    </div>
                `;
            } else {
                statsSummaryHtml = `<span style="color: var(--text-muted); font-size: 11px;">Standard text/ID distribution</span>`;
            }

            // Samples pills
            const samplesHtml = (col.sample_values || [])
                .map((v) => `<span class="sample-chip" title="${escapeHtml(v)}">${escapeHtml(v !== null ? v : 'NULL')}</span>`)
                .join('');

            return `
                <tr>
                    <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${idx + 1}</td>
                    <td>
                        <div class="col-info-cell">
                            <span class="col-name-text">${escapeHtml(col.name)}</span>
                            <div class="col-meta-pills">
                                <span class="type-badge-pill type-${colType}">${colType}</span>
                                <span class="dtype-pill">${escapeHtml(col.pandas_dtype)}</span>
                            </div>
                        </div>
                    </td>
                    <td>
                        <div class="mini-progress-wrap">
                            <div class="mini-progress-labels">
                                <span>${col.missing_percentage}%</span>
                                <span>${col.missing_count} nulls</span>
                            </div>
                            <div class="mini-progress-bg">
                                <div class="mini-progress-bar ${missingBarClass}" style="width: ${Math.max(col.missing_percentage, 0)}%;"></div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <div style="font-size: 12px; font-family: var(--font-mono);">
                            <div><strong>${col.unique_count.toLocaleString()}</strong> unique</div>
                            <div style="color: var(--text-muted); font-size: 11px;">${col.duplicate_count.toLocaleString()} duplicate values</div>
                        </div>
                    </td>
                    <td>
                        <div class="stats-summary-cell">
                            ${statsSummaryHtml}
                        </div>
                    </td>
                    <td>
                        <div class="sample-chips-wrap">
                            ${samplesHtml || '<span class="null-badge">No samples</span>'}
                        </div>
                    </td>
                    <td style="text-align: center;">
                        <button class="btn-inspect" data-col-name="${escapeHtml(col.name)}">
                            Inspect
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        // Attach modal inspector listeners
        document.querySelectorAll('.btn-inspect').forEach((btn) => {
            btn.addEventListener('click', () => {
                const colName = btn.getAttribute('data-col-name');
                openColumnModal(colName);
            });
        });
    }

    // Modal Inspector Renderer
    function openColumnModal(colName) {
        const col = allColumnsProfile.find((c) => c.name === colName);
        if (!col || !modalBackdrop) return;

        const colType = col.classified_type || col.inferred_type || 'Other';

        if (modalColName) modalColName.textContent = col.name;
        if (modalColType) {
            modalColType.textContent = colType;
            modalColType.className = `modal-type-badge type-badge-pill type-${colType}`;
        }
        if (modalColDtype) modalColDtype.textContent = col.pandas_dtype;

        let bodyHtml = `
            <!-- General Counts Grid -->
            <div>
                <h4 style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;">Column Overview</h4>
                <div class="five-number-grid">
                    <div class="summary-stat-box">
                        <div class="stat-box-label">Non-Null Count</div>
                        <div class="stat-box-val">${col.non_null_count.toLocaleString()}</div>
                    </div>
                    <div class="summary-stat-box">
                        <div class="stat-box-label">Missing Count</div>
                        <div class="stat-box-val" style="color: ${col.missing_count > 0 ? 'var(--accent-rose)' : 'var(--text-primary)'};">${col.missing_count.toLocaleString()} (${col.missing_percentage}%)</div>
                    </div>
                    <div class="summary-stat-box">
                        <div class="stat-box-label">Unique Values</div>
                        <div class="stat-box-val">${col.unique_count.toLocaleString()}</div>
                    </div>
                    <div class="summary-stat-box">
                        <div class="stat-box-label">Duplicate Values</div>
                        <div class="stat-box-val">${col.duplicate_count.toLocaleString()}</div>
                    </div>
                </div>
            </div>
        `;

        // Numerical Five-Number & Statistical Summary
        if (colType === 'Numerical' && col.numerical_stats) {
            const ns = col.numerical_stats;
            bodyHtml += `
                <div>
                    <h4 style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;">Five-Number Summary & Moments</h4>
                    <div class="five-number-grid">
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Minimum</div>
                            <div class="stat-box-val">${ns.min}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Q1 (25%)</div>
                            <div class="stat-box-val">${ns.q1}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Median (50%)</div>
                            <div class="stat-box-val" style="color: var(--accent-indigo);">${ns.median}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Q3 (75%)</div>
                            <div class="stat-box-val">${ns.q3}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Maximum</div>
                            <div class="stat-box-val">${ns.max}</div>
                        </div>
                    </div>
                    <div class="five-number-grid" style="margin-top: 10px;">
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Mean (Average)</div>
                            <div class="stat-box-val">${ns.mean}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Std Deviation</div>
                            <div class="stat-box-val">${ns.std}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">IQR</div>
                            <div class="stat-box-val">${ns.iqr}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Skewness</div>
                            <div class="stat-box-val">${ns.skewness}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Zeros Count</div>
                            <div class="stat-box-val">${ns.zeros_count} (${ns.zeros_percentage}%)</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Categorical Frequency Distribution Table
        if (colType === 'Categorical' && col.categorical_stats && col.categorical_stats.top_categories) {
            const topCats = col.categorical_stats.top_categories;
            bodyHtml += `
                <div>
                    <h4 style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;">Top Categories Breakdown (${col.categorical_stats.num_categories} total)</h4>
                    <table class="modal-category-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th style="width: 90px; text-align: right;">Count</th>
                                <th style="width: 90px; text-align: right;">Frequency</th>
                                <th style="width: 140px;">Distribution</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${topCats.map((cat) => `
                                <tr>
                                    <td style="font-weight: 600;">${escapeHtml(cat.category)}</td>
                                    <td style="text-align: right; font-family: var(--font-mono);">${cat.count.toLocaleString()}</td>
                                    <td style="text-align: right; font-family: var(--font-mono);">${cat.percentage}%</td>
                                    <td>
                                        <div class="progress-bar-bg" style="height: 6px;">
                                            <div class="progress-bar-fill fill-indigo" style="width: ${cat.percentage}%;"></div>
                                        </div>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }

        // Datetime Info
        if (colType === 'Datetime' && col.datetime_stats) {
            const ds = col.datetime_stats;
            bodyHtml += `
                <div>
                    <h4 style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;">Temporal Bounds</h4>
                    <div class="five-number-grid">
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Earliest Date</div>
                            <div class="stat-box-val" style="font-size: 13px;">${ds.min_date}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Latest Date</div>
                            <div class="stat-box-val" style="font-size: 13px;">${ds.max_date}</div>
                        </div>
                        <div class="summary-stat-box">
                            <div class="stat-box-label">Span Range</div>
                            <div class="stat-box-val" style="font-size: 13px; color: var(--accent-cyan);">${ds.date_range}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Samples Box
        const sampleChips = (col.sample_values || [])
            .map((v) => `<span class="sample-chip" style="max-width: none; font-size: 12px; padding: 4px 10px;">${escapeHtml(v !== null ? v : 'NULL')}</span>`)
            .join('');

        bodyHtml += `
            <div>
                <h4 style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;">Sample Distinct Observations</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    ${sampleChips || '<span class="null-badge">No samples available</span>'}
                </div>
            </div>
        `;

        if (modalColBody) modalColBody.innerHTML = bodyHtml;
        modalBackdrop.classList.remove('hidden');
    }

    // Modal close events
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => {
            if (modalBackdrop) modalBackdrop.classList.add('hidden');
        });
    }

    if (modalBackdrop) {
        modalBackdrop.addEventListener('click', (e) => {
            if (e.target === modalBackdrop) {
                modalBackdrop.classList.add('hidden');
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modalBackdrop && !modalBackdrop.classList.contains('hidden')) {
            modalBackdrop.classList.add('hidden');
        }
    });

    // 4. Load Paginated Table Preview (Real Data)
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
            tableBody.innerHTML = `<tr><td colspan="${columns.length + 1}" style="text-align: center; color: var(--text-muted); padding: 30px;">No matching records in preview.</td></tr>`;
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

    // Table Search Filter
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

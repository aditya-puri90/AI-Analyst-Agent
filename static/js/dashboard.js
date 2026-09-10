/**
 * Dashboard Client-Side Controller for AI Data Analyst Agent.
 * Phase 4: Automatic Dataset Profiling, Data Quality Auditing, Cleaning Engine & Exploration.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let activeDatasetId = window.INITIAL_DATASET_ID || '';
    let currentProfile = null;
    let allColumnsProfile = [];
    let activeTypeFilter = 'all';
    let columnSearchQuery = '';
    
    // Cleaning Studio State
    let allDetectedIssues = [];
    let activeSeverityFilter = 'all';
    let currentCleaningSummary = null;
    let isViewingCleanedData = false;

    // Explorer Table State
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
    const activeDatasetStatus = document.getElementById('active-dataset-status');
    const openChatBtn = document.getElementById('open-chat-btn');

    // DOM Elements - Studio Tabs
    const studioTabBtns = document.querySelectorAll('.studio-tab-btn');
    const studioTabContents = document.querySelectorAll('.studio-tab-content');
    const cleaningIssuesBadge = document.getElementById('cleaning-issues-badge');

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

    // DOM Elements - Cleaning Studio
    const btnApplyCleaning = document.getElementById('btn-apply-cleaning');
    const btnApplyCleaningBottom = document.getElementById('btn-apply-cleaning-bottom');
    const btnDownloadCleaned = document.getElementById('btn-download-cleaned');
    const cleaningSummarySection = document.getElementById('cleaning-summary-section');
    const cleaningTimestamp = document.getElementById('cleaning-timestamp');
    const cleanKpiRowsBefore = document.getElementById('clean-kpi-rows-before');
    const cleanKpiRowsAfter = document.getElementById('clean-kpi-rows-after');
    const cleanKpiDuplicatesRemoved = document.getElementById('clean-kpi-duplicates-removed');
    const cleanKpiMissingHandled = document.getElementById('clean-kpi-missing-handled');
    const cleanKpiColsConverted = document.getElementById('clean-kpi-cols-converted');
    const cleanKpiValsStandardized = document.getElementById('clean-kpi-vals-standardized');
    const appliedOperationsChips = document.getElementById('applied-operations-chips');

    const sevCountAll = document.getElementById('sev-count-all');
    const sevCountCritical = document.getElementById('sev-count-critical');
    const sevCountHigh = document.getElementById('sev-count-high');
    const sevCountMedium = document.getElementById('sev-count-medium');
    const sevCountLow = document.getElementById('sev-count-low');
    const sevFilterBtns = document.querySelectorAll('.sev-filter-btn');
    const cleaningIssuesTbody = document.getElementById('cleaning-issues-tbody');
    const previewPairsCount = document.getElementById('preview-pairs-count');
    const previewCardsGrid = document.getElementById('preview-cards-grid');

    // Config Checkbox Elements
    const cfgRemoveDuplicates = document.getElementById('cfg-remove-duplicates');
    const cfgFillNumericMissing = document.getElementById('cfg-fill-numeric-missing');
    const cfgNumericStrategy = document.getElementById('cfg-numeric-strategy');
    const cfgFillCategoricalMissing = document.getElementById('cfg-fill-categorical-missing');
    const cfgCategoricalStrategy = document.getElementById('cfg-categorical-strategy');
    const cfgStandardizeWhitespace = document.getElementById('cfg-standardize-whitespace');
    const cfgStandardizeCategorical = document.getElementById('cfg-standardize-categorical');
    const cfgConvertNumericStrings = document.getElementById('cfg-convert-numeric-strings');
    const cfgConvertDatetimeStrings = document.getElementById('cfg-convert-datetime-strings');
    const cfgHandleInvalidNumerical = document.getElementById('cfg-handle-invalid-numerical');
    const cfgDropHighMissing = document.getElementById('cfg-drop-high-missing');
    const cfgDropConstantColumns = document.getElementById('cfg-drop-constant-columns');
    const cfgCapOutliers = document.getElementById('cfg-cap-outliers');

    // DOM Elements - Modal Inspector
    const modalBackdrop = document.getElementById('column-modal-backdrop');
    const modalColName = document.getElementById('modal-col-name');
    const modalColType = document.getElementById('modal-col-type');
    const modalColDtype = document.getElementById('modal-col-dtype');
    const modalColBody = document.getElementById('modal-col-body');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // DOM Elements - Preview Explorer
    const viewRawDataBtn = document.getElementById('view-raw-data-btn');
    const viewCleanDataBtn = document.getElementById('view-clean-data-btn');
    const tableHead = document.getElementById('data-table-head');
    const tableBody = document.getElementById('data-table-body');
    const paginationInfo = document.getElementById('pagination-info');
    const tableSearch = document.getElementById('table-search');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');

    // DOM Elements - Phase 5 Statistical Analysis
    let currentStatistics = null;
    let statsSearchQuery = '';
    const statsKpiNumCount = document.getElementById('stats-kpi-num-count');
    const statsKpiCatCount = document.getElementById('stats-kpi-cat-count');
    const statsKpiDtCount = document.getElementById('stats-kpi-dt-count');
    const statsKpiOutliersCount = document.getElementById('stats-kpi-outliers-count');
    const statsKpiNormalCount = document.getElementById('stats-kpi-normal-count');
    const statsObservationsGrid = document.getElementById('stats-observations-grid');
    const statsTableSearch = document.getElementById('stats-table-search');
    const summaryStatisticsTbody = document.getElementById('summary-statistics-tbody');
    const statsColumnSelector = document.getElementById('stats-column-selector');
    const distributionDetailsCard = document.getElementById('distribution-details-card');

    // DOM Elements - Phase 6 Correlation Analysis
    let currentCorrelation = null;
    let currentHeatmapSpec = null;
    let currentCorrThreshold = 0.0;
    let corrSearchQuery = '';
    const corrThresholdSlider = document.getElementById('corr-threshold-slider');
    const corrThresholdDisplay = document.getElementById('corr-threshold-display');
    const corrKpiNumCount = document.getElementById('corr-kpi-num-count');
    const corrKpiPairsCount = document.getElementById('corr-kpi-pairs-count');
    const corrKpiStrongCount = document.getElementById('corr-kpi-strong-count');
    const corrKpiModCount = document.getElementById('corr-kpi-mod-count');
    const keyCorrelationsGrid = document.getElementById('key-correlations-grid');
    const correlationHeatmapContainer = document.getElementById('correlation-heatmap-container');
    const corrTableSearch = document.getElementById('corr-table-search');
    const rankedCorrelationsTbody = document.getElementById('ranked-correlations-tbody');

    // DOM Elements - Phase 8 Automatic Visualization Engine
    let allRecommendations = [];
    let activeVizFilter = 'all';
    let vizSchema = null;
    const vizCountBadge = document.getElementById('viz-count-badge');
    const vizKpiTotalCount = document.getElementById('viz-kpi-total-count');
    const vizKpiDistCount = document.getElementById('viz-kpi-dist-count');
    const vizKpiRelCount = document.getElementById('viz-kpi-rel-count');
    const vizKpiCatCount = document.getElementById('viz-kpi-cat-count');
    const vizKpiTrendCount = document.getElementById('viz-kpi-trend-count');
    const recFilterAllCount = document.getElementById('rec-filter-all-count');
    const recommendedChartsGrid = document.getElementById('recommended-charts-grid');
    const vizFilterBtns = document.querySelectorAll('.viz-filter-btn');

    // Custom Chart Builder DOM Elements
    const builderXCol = document.getElementById('builder-x-col');
    const builderYCol = document.getElementById('builder-y-col');
    const builderChartType = document.getElementById('builder-chart-type');
    const builderAggregation = document.getElementById('builder-aggregation');
    const builderColorCol = document.getElementById('builder-color-col');
    const builderChartTitle = document.getElementById('builder-chart-title');
    const btnGenerateCustomChart = document.getElementById('btn-generate-custom-chart');
    const btnResetCustomChart = document.getElementById('btn-reset-custom-chart');
    const customPlotlyCanvas = document.getElementById('custom-plotly-canvas');

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

    // Tab Switcher Handling
    studioTabBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            const targetTabId = btn.getAttribute('data-tab');
            studioTabBtns.forEach((b) => b.classList.remove('active'));
            studioTabContents.forEach((c) => {
                c.classList.remove('active');
                c.classList.add('hidden');
            });

            btn.classList.add('active');
            const targetContent = document.getElementById(targetTabId);
            if (targetContent) {
                targetContent.classList.remove('hidden');
                targetContent.classList.add('active');
            }

            // Auto-resize Plotly correlation heatmap when tab becomes visible
            if (targetTabId === 'correlation-view' && window.Plotly && correlationHeatmapContainer) {
                setTimeout(() => {
                    Plotly.Plots.resize(correlationHeatmapContainer);
                }, 50);
            }

            // Auto-resize Plotly outlier chart when tab becomes visible
            if (targetTabId === 'outliers-view' && window.Plotly && document.getElementById('outlier-plotly-chart')) {
                setTimeout(() => {
                    Plotly.Plots.resize('outlier-plotly-chart');
                }, 50);
            }

            // Auto-resize Plotly visualization charts when tab becomes visible
            if (targetTabId === 'visualization-view' && window.Plotly) {
                setTimeout(() => {
                    allRecommendations.forEach((rec) => {
                        const elId = `plotly-rec-${rec.id}`;
                        if (document.getElementById(elId)) {
                            Plotly.Plots.resize(elId);
                        }
                    });
                    if (customPlotlyCanvas) {
                        Plotly.Plots.resize('custom-plotly-canvas');
                    }
                }, 50);
            }
        });
    });

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

        // Update links & download URL
        if (openChatBtn) openChatBtn.href = `/chat?dataset_id=${encodeURIComponent(datasetId)}`;
        if (btnDownloadCleaned) {
            btnDownloadCleaned.href = `/api/cleaning/download/${encodeURIComponent(datasetId)}`;
        }

        try {
            // 1. Fetch Profile
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

            // 2. Fetch Data Quality & Cleaning Audit
            await loadCleaningAudit(datasetId);

            // 3. Fetch Phase 5 Statistical Analysis
            await loadStatisticsData(datasetId);

            // 4. Fetch Phase 6 Correlation Analysis
            await loadCorrelationData(datasetId);

            // 5. Fetch Phase 7 Outlier Detection
            await loadOutlierData(datasetId);

            // 6. Fetch Phase 8 Automatic Visualization Engine
            await loadVisualizationData(datasetId);

            // 7. Load table preview (Raw)
            isViewingCleanedData = false;
            if (viewRawDataBtn) viewRawDataBtn.classList.add('active');
            if (viewCleanDataBtn) viewCleanDataBtn.classList.remove('active');
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
            
            let missingBarClass = 'fill-emerald';
            if (col.missing_percentage > 15) missingBarClass = 'fill-rose';
            else if (col.missing_percentage > 0) missingBarClass = 'fill-amber';

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

    // =========================================================================
    // 4. AUTOMATED DATA QUALITY & CLEANING STUDIO CONTROLLER
    // =========================================================================

    async function loadCleaningAudit(datasetId) {
        try {
            const res = await fetch(`/api/cleaning/audit/${encodeURIComponent(datasetId)}`);
            const data = await res.json();

            if (!res.ok || !data.success) {
                console.error('Failed to load cleaning audit:', data.error);
                return;
            }

            allDetectedIssues = data.issues || [];
            const sc = data.severity_counts || { Critical: 0, High: 0, Medium: 0, Low: 0 };

            // Update badge counters
            if (cleaningIssuesBadge) cleaningIssuesBadge.textContent = allDetectedIssues.length;
            if (sevCountAll) sevCountAll.textContent = allDetectedIssues.length;
            if (sevCountCritical) sevCountCritical.textContent = sc.Critical || 0;
            if (sevCountHigh) sevCountHigh.textContent = sc.High || 0;
            if (sevCountMedium) sevCountMedium.textContent = sc.Medium || 0;
            if (sevCountLow) sevCountLow.textContent = sc.Low || 0;

            renderCleaningIssuesTable();
            renderTransformationPreviews(data.preview);

        } catch (e) {
            console.error('Error fetching cleaning audit:', e);
        }
    }

    // Severity Filter buttons handler
    sevFilterBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            sevFilterBtns.forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            activeSeverityFilter = btn.getAttribute('data-sev');
            renderCleaningIssuesTable();
        });
    });

    // Render 12-point Quality Issues Table
    function renderCleaningIssuesTable() {
        if (!cleaningIssuesTbody) return;

        let filtered = allDetectedIssues;
        if (activeSeverityFilter !== 'all') {
            filtered = filtered.filter((i) => i.severity.toLowerCase() === activeSeverityFilter.toLowerCase());
        }

        if (filtered.length === 0) {
            cleaningIssuesTbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--accent-emerald); padding: 30px; font-weight: 600;">✨ No data quality issues detected for the selected filter!</td></tr>`;
            return;
        }

        cleaningIssuesTbody.innerHTML = filtered.map((issue, idx) => {
            const sev = issue.severity || 'Low';
            const sevClass = `sev-badge sev-badge-${sev.toLowerCase()}`;
            const colDisplay = issue.column === '(Entire Dataset)' 
                ? `<span style="font-weight: 700; color: var(--accent-indigo);">(Entire Dataset)</span>`
                : `<strong style="color: var(--text-primary); font-family: var(--font-mono);">${escapeHtml(issue.column)}</strong>`;

            return `
                <tr>
                    <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${idx + 1}</td>
                    <td>${colDisplay}</td>
                    <td>
                        <div style="font-weight: 600; color: var(--text-primary); font-size: 13px;">${escapeHtml(issue.title)}</div>
                        <div style="color: var(--text-muted); font-size: 11px; font-family: var(--font-mono);">${escapeHtml(issue.issue_type)}</div>
                    </td>
                    <td>
                        <div style="font-size: 12px; font-family: var(--font-mono);">
                            <strong>${issue.affected_rows.toLocaleString()}</strong> rows
                            <span style="color: var(--text-muted);">(${issue.percentage_affected}%)</span>
                        </div>
                    </td>
                    <td>
                        <span class="${sevClass}">${sev}</span>
                    </td>
                    <td>
                        <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.4;">
                            ${escapeHtml(issue.recommended_action)}
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }

    // Render Transformation Preview Cards (Original -> Cleaned)
    function renderTransformationPreviews(previewData) {
        if (!previewCardsGrid) return;

        const previews = previewData && previewData.previews ? previewData.previews : [];
        if (previewPairsCount) previewPairsCount.textContent = `${previews.length} proposed transformation${previews.length === 1 ? '' : 's'}`;

        if (previews.length === 0) {
            previewCardsGrid.innerHTML = `
                <div class="preview-card-placeholder">
                    ✨ No transformation previews needed for this dataset. All data conforms to standard formatting.
                </div>
            `;
            return;
        }

        previewCardsGrid.innerHTML = previews.map((p) => {
            return `
                <div class="preview-card">
                    <div class="preview-card-header">
                        <span class="preview-card-col">${escapeHtml(p.column)}</span>
                        <span class="preview-card-category">${escapeHtml(p.category)}</span>
                    </div>
                    <div class="preview-transform-flow">
                        <span class="preview-orig" title="Original raw value">${escapeHtml(p.original_value)}</span>
                        <span class="preview-arrow">&rarr;</span>
                        <span class="preview-clean" title="Proposed cleaned value">${escapeHtml(p.cleaned_value)}</span>
                    </div>
                    <div class="preview-rule">${escapeHtml(p.rule)}</div>
                </div>
            `;
        }).join('');
    }

    // Gather configurable cleaning operations from checkboxes
    function getSelectedCleaningOperations() {
        return {
            remove_duplicates: cfgRemoveDuplicates ? cfgRemoveDuplicates.checked : true,
            fill_numeric_missing: cfgFillNumericMissing && cfgFillNumericMissing.checked 
                ? (cfgNumericStrategy ? cfgNumericStrategy.value : 'median') 
                : 'none',
            fill_categorical_missing: cfgFillCategoricalMissing && cfgFillCategoricalMissing.checked 
                ? (cfgCategoricalStrategy ? cfgCategoricalStrategy.value : 'mode') 
                : 'none',
            standardize_whitespace: cfgStandardizeWhitespace ? cfgStandardizeWhitespace.checked : true,
            standardize_categorical: cfgStandardizeCategorical ? cfgStandardizeCategorical.checked : true,
            convert_numeric_strings: cfgConvertNumericStrings ? cfgConvertNumericStrings.checked : true,
            convert_datetime_strings: cfgConvertDatetimeStrings ? cfgConvertDatetimeStrings.checked : true,
            handle_invalid_numerical: cfgHandleInvalidNumerical ? cfgHandleInvalidNumerical.checked : true,
            drop_high_missing_columns: cfgDropHighMissing ? cfgDropHighMissing.checked : false,
            drop_constant_columns: cfgDropConstantColumns ? cfgDropConstantColumns.checked : false,
            cap_outliers: cfgCapOutliers ? cfgCapOutliers.checked : false,
        };
    }

    // Apply Cleaning Pipeline Action Handler
    async function handleApplyCleaning() {
        if (!activeDatasetId) return;

        const originalBtnHtml = btnApplyCleaning ? btnApplyCleaning.innerHTML : '';
        if (btnApplyCleaning) {
            btnApplyCleaning.disabled = true;
            btnApplyCleaning.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin: 0; display: inline-block;"></span> Applying...`;
        }
        if (btnApplyCleaningBottom) {
            btnApplyCleaningBottom.disabled = true;
            btnApplyCleaningBottom.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin: 0; display: inline-block;"></span> Applying...`;
        }

        const operations = getSelectedCleaningOperations();

        try {
            const res = await fetch(`/api/cleaning/apply/${encodeURIComponent(activeDatasetId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ operations }),
            });

            const data = await res.json();

            if (!res.ok || !data.success) {
                alert(`Cleaning failed: ${data.error || 'Unknown error'}`);
                return;
            }

            currentCleaningSummary = data.summary;
            renderCleaningSummary(data.summary);

            // Update download button
            if (btnDownloadCleaned) {
                btnDownloadCleaned.href = data.download_url || `/api/cleaning/download/${encodeURIComponent(activeDatasetId)}`;
            }

            // Show active status tag
            if (activeDatasetStatus) activeDatasetStatus.classList.remove('hidden');

            // Refresh audit findings after cleaning
            await loadCleaningAudit(activeDatasetId);

            // Switch explorer to cleaned preview
            if (viewCleanDataBtn) viewCleanDataBtn.classList.remove('hidden');

            // Scroll to summary smoothly
            if (cleaningSummarySection) {
                cleaningSummarySection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

        } catch (err) {
            alert(`Error applying cleaning pipeline: ${err.message}`);
        } finally {
            if (btnApplyCleaning) {
                btnApplyCleaning.disabled = false;
                btnApplyCleaning.innerHTML = originalBtnHtml;
            }
            if (btnApplyCleaningBottom) {
                btnApplyCleaningBottom.disabled = false;
                btnApplyCleaningBottom.innerHTML = `<span class="btn-icon">⚡</span><span>Apply Cleaning Pipeline</span>`;
            }
        }
    }

    // Attach Apply Button Listeners
    if (btnApplyCleaning) btnApplyCleaning.addEventListener('click', handleApplyCleaning);
    if (btnApplyCleaningBottom) btnApplyCleaningBottom.addEventListener('click', handleApplyCleaning);

    // Render Cleaning Summary Box
    function renderCleaningSummary(summary) {
        if (!summary || !cleaningSummarySection) return;

        cleaningSummarySection.classList.remove('hidden');
        if (cleaningTimestamp) cleaningTimestamp.textContent = new Date().toLocaleTimeString();

        if (cleanKpiRowsBefore) cleanKpiRowsBefore.textContent = summary.rows_before.toLocaleString();
        if (cleanKpiRowsAfter) cleanKpiRowsAfter.textContent = summary.rows_after.toLocaleString();
        if (cleanKpiDuplicatesRemoved) cleanKpiDuplicatesRemoved.textContent = summary.duplicates_removed.toLocaleString();
        if (cleanKpiMissingHandled) cleanKpiMissingHandled.textContent = summary.missing_values_handled.toLocaleString();
        if (cleanKpiColsConverted) cleanKpiColsConverted.textContent = summary.columns_converted.toLocaleString();
        if (cleanKpiValsStandardized) cleanKpiValsStandardized.textContent = summary.values_standardized.toLocaleString();

        if (appliedOperationsChips) {
            appliedOperationsChips.innerHTML = '';
            (summary.operations_applied || []).forEach((op) => {
                const chip = document.createElement('span');
                chip.className = 'applied-chip';
                chip.textContent = `✓ ${op}`;
                appliedOperationsChips.appendChild(chip);
            });
            if (!summary.operations_applied || summary.operations_applied.length === 0) {
                appliedOperationsChips.innerHTML = `<span style="color: var(--text-muted); font-size: 12px;">No transformations needed (dataset already clean).</span>`;
            }
        }
    }

    // Explorer Raw vs Clean Toggle
    if (viewRawDataBtn) {
        viewRawDataBtn.addEventListener('click', async () => {
            isViewingCleanedData = false;
            viewRawDataBtn.classList.add('active');
            if (viewCleanDataBtn) viewCleanDataBtn.classList.remove('active');
            currentPage = 1;
            await loadTablePreview(activeDatasetId, currentPage);
        });
    }

    if (viewCleanDataBtn) {
        viewCleanDataBtn.addEventListener('click', async () => {
            isViewingCleanedData = true;
            viewCleanDataBtn.classList.add('active');
            if (viewRawDataBtn) viewRawDataBtn.classList.remove('active');
            currentPage = 1;
            await loadTablePreview(activeDatasetId, currentPage);
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

    // 5. Load Paginated Table Preview (Real Data)
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

        tableHead.innerHTML = `
            <tr>
                <th style="width: 40px;">#</th>
                ${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}
            </tr>
        `;

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

    // =========================================================================
    // PHASE 5: STATISTICAL ANALYSIS ENGINE CONTROLLER
    // =========================================================================
    async function loadStatisticsData(datasetId) {
        try {
            const res = await fetch(`/api/statistics/${encodeURIComponent(datasetId)}`);
            const data = await res.json();
            if (res.ok && data.success && data.statistics) {
                currentStatistics = data.statistics;
                renderStatisticalAnalysis(currentStatistics);
            }
        } catch (e) {
            console.error('Failed to load statistical analysis:', e);
        }
    }

    function renderStatisticalAnalysis(statsData) {
        if (!statsData) return;

        // 1. Render Quick KPIs
        const numStats = statsData.numerical_statistics || {};
        const catStats = statsData.categorical_statistics || {};
        const dtStats = statsData.datetime_statistics || {};
        const obs = statsData.statistical_observations || [];

        const numCount = Object.keys(numStats).length;
        const catCount = Object.keys(catStats).length;
        const dtCount = Object.keys(dtStats).length;

        // Count outlier features
        let outlierFeatures = 0;
        let normalCount = 0;
        for (const [colName, n] of Object.entries(numStats)) {
            if (n.outliers && n.outliers.count > 0) outlierFeatures++;
            if (n.normality && n.normality.is_normal) normalCount++;
        }

        if (statsKpiNumCount) statsKpiNumCount.textContent = numCount;
        if (statsKpiCatCount) statsKpiCatCount.textContent = catCount;
        if (statsKpiDtCount) statsKpiDtCount.textContent = dtCount;
        if (statsKpiOutliersCount) statsKpiOutliersCount.textContent = `${outlierFeatures} features with outliers`;
        if (statsKpiNormalCount) statsKpiNormalCount.textContent = `${normalCount} of ${numCount} Gaussian`;

        // 2. Render Statistical Observations
        renderStatisticalObservations(obs);

        // 3. Render Summary Statistics Table
        renderSummaryStatisticsTable(numStats);

        // 4. Populate Feature Selector for Distribution Deep Dive
        if (statsColumnSelector) {
            statsColumnSelector.innerHTML = '<option value="">-- Choose a Feature to Inspect --</option>';

            // Numerical group
            if (numCount > 0) {
                const numGroup = document.createElement('optgroup');
                numGroup.label = '🔢 Numerical Features';
                for (const colName of Object.keys(numStats)) {
                    const opt = document.createElement('option');
                    opt.value = colName;
                    opt.textContent = colName;
                    numGroup.appendChild(opt);
                }
                statsColumnSelector.appendChild(numGroup);
            }

            // Categorical group
            if (catCount > 0) {
                const catGroup = document.createElement('optgroup');
                catGroup.label = '🏷️ Categorical Features';
                for (const colName of Object.keys(catStats)) {
                    const opt = document.createElement('option');
                    opt.value = colName;
                    opt.textContent = colName;
                    catGroup.appendChild(opt);
                }
                statsColumnSelector.appendChild(catGroup);
            }

            // Datetime group
            if (dtCount > 0) {
                const dtGroup = document.createElement('optgroup');
                dtGroup.label = '📅 Datetime Features';
                for (const colName of Object.keys(dtStats)) {
                    const opt = document.createElement('option');
                    opt.value = colName;
                    opt.textContent = colName;
                    dtGroup.appendChild(opt);
                }
                statsColumnSelector.appendChild(dtGroup);
            }

            // Default select first numerical column
            const firstNumCol = Object.keys(numStats)[0] || Object.keys(catStats)[0] || Object.keys(dtStats)[0];
            if (firstNumCol) {
                statsColumnSelector.value = firstNumCol;
                renderDistributionDeepDive(statsData, firstNumCol);
            }
        }
    }

    function renderStatisticalObservations(observations) {
        if (!statsObservationsGrid) return;
        if (!observations || observations.length === 0) {
            statsObservationsGrid.innerHTML = `
                <div class="stats-obs-card info">
                    <div class="obs-card-header">
                        <span class="obs-icon">✨</span>
                        <h4 class="obs-title">Clean Statistical Balance</h4>
                    </div>
                    <p class="obs-message">No critical distributional defects, heavy outliers, or extreme skewness were detected.</p>
                </div>
            `;
            return;
        }

        const icons = {
            'Quality Warning': '⚠️',
            'Zero Variance': '⚪',
            'Distribution Shape': '📊',
            'Heavy Tails': '📐',
            'Outlier Alert': '🚨',
            'High Dispersion': '📈',
            'Class Imbalance': '⚖️',
            'High Cardinality': '🏷️',
            'Temporal Span': '📅',
            'General Quality': '✨',
        };

        statsObservationsGrid.innerHTML = observations.map((obs) => {
            const icon = icons[obs.type] || '🔍';
            const cardClass = obs.severity === 'high' ? 'high' : (obs.severity === 'warning' ? 'warning' : 'info');
            return `
                <div class="stats-obs-card ${cardClass}">
                    <div class="obs-card-header">
                        <span class="obs-icon">${icon}</span>
                        <h4 class="obs-title">${escapeHtml(obs.title)}</h4>
                    </div>
                    <p class="obs-message">${escapeHtml(obs.message)}</p>
                </div>
            `;
        }).join('');
    }

    function renderSummaryStatisticsTable(numStats) {
        if (!summaryStatisticsTbody) return;

        let entries = Object.entries(numStats || {});

        if (statsSearchQuery) {
            entries = entries.filter(([colName]) => colName.toLowerCase().includes(statsSearchQuery));
        }

        if (entries.length === 0) {
            summaryStatisticsTbody.innerHTML = `<tr><td colspan="15" style="text-align: center; color: var(--text-muted); padding: 30px;">No numerical features matching '${escapeHtml(statsSearchQuery)}'</td></tr>`;
            return;
        }

        summaryStatisticsTbody.innerHTML = entries.map(([colName, n], idx) => {
            const ci = n.confidence_interval || {};
            const outliers = n.outliers || {};
            const ciStr = ci.lower !== null && ci.upper !== null ? `[${ci.lower}, ${ci.upper}]` : '--';
            const rangeStr = n.min !== null && n.max !== null ? `[${n.min}, ${n.max}]` : '--';
            const iqrBounds = n.q1 !== null && n.q3 !== null ? `${n.iqr} [${n.q1}, ${n.q3}]` : '--';

            // Skewness pill styling
            let skewPill = `<span class="skew-pill skew-normal">${n.skewness !== null ? n.skewness : '--'}</span>`;
            if (n.skewness !== null && n.skewness > 1.0) {
                skewPill = `<span class="skew-pill skew-positive" title="Right Skewed">${n.skewness} ↗</span>`;
            } else if (n.skewness !== null && n.skewness < -1.0) {
                skewPill = `<span class="skew-pill skew-negative" title="Left Skewed">${n.skewness} ↖</span>`;
            }

            // Outlier styling
            const outlierText = outliers.count > 0 
                ? `<span style="color: var(--accent-rose); font-weight: 700;">${outliers.count} (${outliers.percentage}%)</span>`
                : `<span style="color: var(--accent-emerald);">0</span>`;

            return `
                <tr>
                    <td style="color: var(--text-muted);">${idx + 1}</td>
                    <td class="stats-col-name">${escapeHtml(colName)}</td>
                    <td>${n.valid_count !== undefined ? n.valid_count.toLocaleString() : '--'}</td>
                    <td style="font-weight: 700; color: var(--accent-indigo);">${n.mean !== null ? n.mean : '--'}</td>
                    <td>${n.median !== null ? n.median : '--'}</td>
                    <td style="color: var(--text-secondary);">${n.mode !== null ? n.mode : '--'}</td>
                    <td>${n.std !== null ? n.std : '--'}</td>
                    <td>${n.variance !== null ? n.variance : '--'}</td>
                    <td>${rangeStr}</td>
                    <td>${n.range !== null ? n.range : '--'}</td>
                    <td>${iqrBounds}</td>
                    <td>${skewPill}</td>
                    <td>${n.kurtosis !== null ? n.kurtosis : '--'}</td>
                    <td><span class="ci-pill">${ciStr}</span></td>
                    <td>${outlierText}</td>
                </tr>
            `;
        }).join('');
    }

    function renderDistributionDeepDive(statsData, colName) {
        if (!distributionDetailsCard || !colName) return;

        const numStats = statsData.numerical_statistics || {};
        const catStats = statsData.categorical_statistics || {};
        const dtStats = statsData.datetime_statistics || {};

        if (numStats[colName]) {
            const n = numStats[colName];
            const p = n.percentiles || {};
            const ci = n.confidence_interval || {};
            const norm = n.normality || {};
            const disp = n.dispersion_metrics || {};
            const outliers = n.outliers || {};

            let normBadgeClass = 'normality-normal';
            if (norm.distribution_shape && norm.distribution_shape.includes('Skewed')) normBadgeClass = 'normality-skewed';
            if (norm.distribution_shape && norm.distribution_shape.includes('Heavy')) normBadgeClass = 'normality-heavy';

            distributionDetailsCard.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <h4 style="font-size: 18px; font-weight: 800; color: var(--text-primary);">${escapeHtml(colName)}</h4>
                        <span style="font-size: 12px; color: var(--text-muted);">Numerical Feature • ${n.valid_count.toLocaleString()} valid observations (${n.missing_percentage}% missing)</span>
                    </div>
                    <div>
                        <span class="normality-badge ${normBadgeClass}">
                            <span>●</span>
                            <span>${norm.distribution_shape || 'Distribution Analysis'}</span>
                        </span>
                    </div>
                </div>

                <div class="dist-grid-layout">
                    <!-- Percentiles Matrix -->
                    <div class="dist-box">
                        <div class="dist-box-title">Percentiles Spectrum (P1 – P99)</div>
                        <div class="percentiles-grid">
                            <div class="percentile-chip">
                                <div class="percentile-label">P1 (Min Tail)</div>
                                <div class="percentile-val">${p.p1 !== undefined ? p.p1 : '--'}</div>
                            </div>
                            <div class="percentile-chip">
                                <div class="percentile-label">P5</div>
                                <div class="percentile-val">${p.p5 !== undefined ? p.p5 : '--'}</div>
                            </div>
                            <div class="percentile-chip">
                                <div class="percentile-label">P10</div>
                                <div class="percentile-val">${p.p10 !== undefined ? p.p10 : '--'}</div>
                            </div>
                            <div class="percentile-chip" style="background: rgba(99, 102, 241, 0.12); border-color: rgba(99, 102, 241, 0.3);">
                                <div class="percentile-label" style="color: var(--accent-indigo);">P25 (Q1)</div>
                                <div class="percentile-val" style="color: var(--accent-indigo);">${p.p25 !== undefined ? p.p25 : '--'}</div>
                            </div>
                            <div class="percentile-chip" style="background: rgba(99, 102, 241, 0.2); border-color: rgba(99, 102, 241, 0.4);">
                                <div class="percentile-label" style="color: #fff;">P50 (Median)</div>
                                <div class="percentile-val" style="color: #fff;">${p.p50 !== undefined ? p.p50 : '--'}</div>
                            </div>
                            <div class="percentile-chip" style="background: rgba(99, 102, 241, 0.12); border-color: rgba(99, 102, 241, 0.3);">
                                <div class="percentile-label" style="color: var(--accent-indigo);">P75 (Q3)</div>
                                <div class="percentile-val" style="color: var(--accent-indigo);">${p.p75 !== undefined ? p.p75 : '--'}</div>
                            </div>
                            <div class="percentile-chip">
                                <div class="percentile-label">P90</div>
                                <div class="percentile-val">${p.p90 !== undefined ? p.p90 : '--'}</div>
                            </div>
                            <div class="percentile-chip">
                                <div class="percentile-label">P95</div>
                                <div class="percentile-val">${p.p95 !== undefined ? p.p95 : '--'}</div>
                            </div>
                            <div class="percentile-chip">
                                <div class="percentile-label">P99 (Max Tail)</div>
                                <div class="percentile-val">${p.p99 !== undefined ? p.p99 : '--'}</div>
                            </div>
                        </div>
                    </div>

                    <!-- Dispersion & Confidence Interval -->
                    <div class="dist-box">
                        <div class="dist-box-title">Dispersion & Confidence Bounds</div>
                        <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13px;">
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">95% Confidence Interval for Mean:</span>
                                <strong style="font-family: var(--font-mono); color: var(--accent-indigo);">[${ci.lower}, ${ci.upper}]</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">Standard Error (SEM):</span>
                                <strong style="font-family: var(--font-mono);">${disp.sem !== null ? disp.sem : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">Coefficient of Variation (CV):</span>
                                <strong style="font-family: var(--font-mono);">${disp.cv_percentage !== null ? disp.cv_percentage + '%' : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">Fisher Excess Kurtosis:</span>
                                <strong style="font-family: var(--font-mono);">${n.kurtosis !== null ? n.kurtosis : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-secondary);">Fisher-Pearson Skewness:</span>
                                <strong style="font-family: var(--font-mono);">${n.skewness !== null ? n.skewness : '--'}</strong>
                            </div>
                        </div>
                    </div>

                    <!-- Outlier Analysis Box -->
                    <div class="dist-box">
                        <div class="dist-box-title">1.5x IQR Outlier Detection</div>
                        <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13px;">
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">Lower Bound (Q1 - 1.5*IQR):</span>
                                <strong style="font-family: var(--font-mono);">${outliers.lower_bound !== null ? outliers.lower_bound : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">Upper Bound (Q3 + 1.5*IQR):</span>
                                <strong style="font-family: var(--font-mono);">${outliers.upper_bound !== null ? outliers.upper_bound : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px;">
                                <span style="color: var(--text-secondary);">Detected Outliers:</span>
                                <strong style="font-family: var(--font-mono); color: ${outliers.count > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">${outliers.count} (${outliers.percentage}%)</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-secondary);">Zeros Count:</span>
                                <strong style="font-family: var(--font-mono);">${disp.zeros_count} (${disp.zeros_percentage}%)</strong>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else if (catStats[colName]) {
            const c = catStats[colName];
            const topCats = c.top_categories || [];

            distributionDetailsCard.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <h4 style="font-size: 18px; font-weight: 800; color: var(--text-primary);">${escapeHtml(colName)}</h4>
                        <span style="font-size: 12px; color: var(--text-muted);">Categorical Feature • ${c.valid_count.toLocaleString()} valid values • ${c.num_categories} distinct categories</span>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <span class="meta-tag">Shannon Entropy: ${c.entropy !== null ? c.entropy : '--'}</span>
                        <span class="meta-tag">Rare (<1%): ${c.rare_categories_count}</span>
                    </div>
                </div>

                <div class="dist-box">
                    <div class="dist-box-title">Category Percentage Distribution & Cumulative Share</div>
                    <table class="modal-category-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th style="width: 90px; text-align: right;">Count</th>
                                <th style="width: 90px; text-align: right;">Frequency</th>
                                <th style="width: 100px; text-align: right;">Cumulative</th>
                                <th style="width: 160px;">Share</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${topCats.map((cat) => `
                                <tr>
                                    <td style="font-weight: 700; color: var(--text-primary);">${escapeHtml(cat.category)}</td>
                                    <td style="text-align: right; font-family: var(--font-mono);">${cat.count.toLocaleString()}</td>
                                    <td style="text-align: right; font-family: var(--font-mono);">${cat.percentage}%</td>
                                    <td style="text-align: right; font-family: var(--font-mono); color: var(--accent-cyan);">${cat.cumulative_percentage}%</td>
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
        } else if (dtStats[colName]) {
            const d = dtStats[colName];
            const months = d.records_by_month || {};
            const dows = d.records_by_day_of_week || {};

            distributionDetailsCard.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <h4 style="font-size: 18px; font-weight: 800; color: var(--text-primary);">${escapeHtml(colName)}</h4>
                        <span style="font-size: 12px; color: var(--text-muted);">Datetime Feature • ${d.valid_count.toLocaleString()} valid dates • Inferred Cadence: ${d.inferred_frequency}</span>
                    </div>
                    <div>
                        <span class="meta-tag meta-tag-accent">Span: ${d.date_range_formatted}</span>
                    </div>
                </div>

                <div class="dist-grid-layout">
                    <div class="dist-box">
                        <div class="dist-box-title">Date Span & Boundaries</div>
                        <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-secondary);">Earliest Date:</span>
                                <strong style="font-family: var(--font-mono);">${d.min_date ? d.min_date.split('T')[0] : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-secondary);">Latest Date:</span>
                                <strong style="font-family: var(--font-mono);">${d.max_date ? d.max_date.split('T')[0] : '--'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-secondary);">Total Elapsed Days:</span>
                                <strong style="font-family: var(--font-mono); color: var(--accent-cyan);">${d.date_range_days.toLocaleString()} days</strong>
                            </div>
                        </div>
                    </div>

                    <div class="dist-box">
                        <div class="dist-box-title">Records by Month</div>
                        <div style="display: flex; flex-direction: column; gap: 6px; max-height: 180px; overflow-y: auto;">
                            ${Object.entries(months).map(([m, cnt]) => `
                                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                                    <span>${m}</span>
                                    <strong style="font-family: var(--font-mono);">${cnt.toLocaleString()}</strong>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <div class="dist-box">
                        <div class="dist-box-title">Records by Day of Week</div>
                        <div style="display: flex; flex-direction: column; gap: 6px; max-height: 180px; overflow-y: auto;">
                            ${Object.entries(dows).map(([dow, cnt]) => `
                                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                                    <span>${dow}</span>
                                    <strong style="font-family: var(--font-mono);">${cnt.toLocaleString()}</strong>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        }
    }

    // Stats Table Search Handler
    if (statsTableSearch) {
        statsTableSearch.addEventListener('input', (e) => {
            statsSearchQuery = e.target.value.toLowerCase().trim();
            if (currentStatistics && currentStatistics.numerical_statistics) {
                renderSummaryStatisticsTable(currentStatistics.numerical_statistics);
            }
        });
    }

    // Feature Selector Change Handler
    if (statsColumnSelector) {
        statsColumnSelector.addEventListener('change', (e) => {
            const selectedCol = e.target.value;
            if (selectedCol && currentStatistics) {
                renderDistributionDeepDive(currentStatistics, selectedCol);
            }
        });
    }

    // =========================================================================
    // PHASE 6: CORRELATION ANALYSIS ENGINE CONTROLLER
    // =========================================================================
    async function loadCorrelationData(datasetId, threshold = 0.0) {
        try {
            const res = await fetch(`/api/correlation/${encodeURIComponent(datasetId)}?threshold=${threshold}`);
            const data = await res.json();
            if (res.ok && data.success && data.correlation) {
                currentCorrelation = data.correlation;
                currentHeatmapSpec = data.heatmap_spec;
                renderCorrelationAnalysis(currentCorrelation, currentHeatmapSpec);
            }
        } catch (e) {
            console.error('Failed to load correlation analysis:', e);
        }
    }

    function renderCorrelationAnalysis(corrData, heatmapSpec) {
        if (!corrData) return;

        // 1. Render KPIs
        const numCount = corrData.total_numerical_columns || 0;
        const totalPairs = corrData.total_pairs_count || 0;
        const strCounts = corrData.strength_counts || {};
        const strongCount = (strCounts.very_strong || 0) + (strCounts.strong || 0);
        const modCount = strCounts.moderate || 0;

        if (corrKpiNumCount) corrKpiNumCount.textContent = numCount;
        if (corrKpiPairsCount) corrKpiPairsCount.textContent = totalPairs;
        if (corrKpiStrongCount) corrKpiStrongCount.textContent = `${strongCount} pairs`;
        if (corrKpiModCount) corrKpiModCount.textContent = `${modCount} pairs`;

        // 2. Render Key Correlations
        renderKeyCorrelations(corrData.key_correlations);

        // 3. Render Plotly Heatmap
        if (heatmapSpec) {
            renderCorrelationHeatmap(heatmapSpec);
        }

        // 4. Render Ranked Associations Table
        renderRankedCorrelationsTable(corrData.all_ranked_pairs || corrData.ranked_pairs || []);
    }

    function renderKeyCorrelations(keyCorrs) {
        if (!keyCorrelationsGrid) return;
        const highlights = keyCorrs && keyCorrs.key_highlights ? keyCorrs.key_highlights : [];

        if (highlights.length === 0) {
            keyCorrelationsGrid.innerHTML = `
                <div class="key-corr-empty">
                    <p>No significant linear correlations (|r| &ge; 0.20) detected between numerical features in this dataset.</p>
                </div>
            `;
            return;
        }

        keyCorrelationsGrid.innerHTML = highlights.map((h) => {
            const isPos = h.direction === 'Positive';
            const cardClass = isPos ? 'positive' : 'negative';
            const dirIcon = isPos ? '↗' : '↘';
            const rVal = h.correlation !== null ? Number(h.correlation).toFixed(4) : '--';

            // Strength Badge Class
            let strClass = 'strength-very-weak';
            if (h.strength === 'Very strong') strClass = 'strength-very-strong';
            else if (h.strength === 'Strong') strClass = 'strength-strong';
            else if (h.strength === 'Moderate') strClass = 'strength-moderate';
            else if (h.strength === 'Weak') strClass = 'strength-weak';

            return `
                <div class="key-corr-card ${cardClass}">
                    <div class="key-corr-header">
                        <div class="key-corr-pair-title">
                            <span>${escapeHtml(h.variable_a)}</span>
                            <span style="color: var(--text-muted); font-size: 13px;">↔</span>
                            <span>${escapeHtml(h.variable_b)}</span>
                        </div>
                        <span class="key-corr-r-pill">r = ${rVal}</span>
                    </div>
                    <div class="key-corr-badges-row">
                        <span class="direction-pill ${isPos ? 'direction-positive' : 'direction-negative'}">
                            <span>${dirIcon}</span>
                            <span>${escapeHtml(h.direction)}</span>
                        </span>
                        <span class="strength-pill ${strClass}">
                            <span>${escapeHtml(h.strength)}</span>
                        </span>
                    </div>
                    <div class="key-corr-explanation">
                        ${escapeHtml(h.explanation)}
                    </div>
                </div>
            `;
        }).join('');
    }

    function renderCorrelationHeatmap(spec) {
        if (!correlationHeatmapContainer || !spec) return;

        if (!window.Plotly) {
            correlationHeatmapContainer.innerHTML = `
                <div class="heatmap-loading-placeholder">
                    <p style="color: var(--accent-rose);">Plotly.js library could not be loaded. Please check network connection.</p>
                </div>
            `;
            return;
        }

        const data = spec.data || [];
        const layout = spec.layout || {};
        const config = spec.config || { responsive: true };

        Plotly.react('correlation-heatmap-container', data, layout, config);
    }

    function renderRankedCorrelationsTable(pairs) {
        if (!rankedCorrelationsTbody) return;

        let filtered = pairs || [];

        // Apply threshold filter
        if (currentCorrThreshold > 0) {
            filtered = filtered.filter((p) => {
                const absR = p.absolute_correlation !== undefined ? p.absolute_correlation : Math.abs(p.correlation || 0);
                return absR >= currentCorrThreshold;
            });
        }

        // Apply search query filter
        if (corrSearchQuery) {
            filtered = filtered.filter((p) => {
                const a = (p.variable_a || '').toLowerCase();
                const b = (p.variable_b || '').toLowerCase();
                const str = (p.strength || '').toLowerCase();
                const dir = (p.direction || '').toLowerCase();
                return a.includes(corrSearchQuery) || b.includes(corrSearchQuery) || str.includes(corrSearchQuery) || dir.includes(corrSearchQuery);
            });
        }

        if (filtered.length === 0) {
            rankedCorrelationsTbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 30px;">
                        No correlation pairs matching threshold |r| &ge; ${currentCorrThreshold.toFixed(2)} ${corrSearchQuery ? `and query '${escapeHtml(corrSearchQuery)}'` : ''}
                    </td>
                </tr>
            `;
            return;
        }

        rankedCorrelationsTbody.innerHTML = filtered.map((p, idx) => {
            const rVal = p.correlation !== null ? Number(p.correlation).toFixed(4) : '--';
            const isPos = p.direction === 'Positive';
            const isNeg = p.direction === 'Negative';
            const dirIcon = isPos ? '↗' : (isNeg ? '↘' : '—');
            const dirClass = isPos ? 'direction-positive' : (isNeg ? 'direction-negative' : 'direction-neutral');

            let strClass = 'strength-very-weak';
            if (p.strength === 'Very strong') strClass = 'strength-very-strong';
            else if (p.strength === 'Strong') strClass = 'strength-strong';
            else if (p.strength === 'Moderate') strClass = 'strength-moderate';
            else if (p.strength === 'Weak') strClass = 'strength-weak';

            // Color highlight for r value
            let rColor = 'var(--text-primary)';
            if (isPos && Math.abs(p.correlation) >= 0.6) rColor = 'var(--accent-emerald)';
            else if (isNeg && Math.abs(p.correlation) >= 0.6) rColor = 'var(--accent-rose)';

            return `
                <tr>
                    <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${idx + 1}</td>
                    <td style="font-weight: 700; color: var(--text-primary);">${escapeHtml(p.variable_a)}</td>
                    <td style="font-weight: 700; color: var(--text-primary);">${escapeHtml(p.variable_b)}</td>
                    <td><strong style="font-family: var(--font-mono); color: ${rColor};">${rVal}</strong></td>
                    <td><span class="strength-pill ${strClass}">${escapeHtml(p.strength)}</span></td>
                    <td>
                        <span class="direction-pill ${dirClass}">
                            <span>${dirIcon}</span>
                            <span>${escapeHtml(p.direction)}</span>
                        </span>
                    </td>
                    <td style="font-family: var(--font-mono); color: var(--text-secondary);">${p.sample_size !== undefined ? p.sample_size.toLocaleString() : '--'}</td>
                </tr>
            `;
        }).join('');
    }

    // Threshold Slider Listener
    if (corrThresholdSlider) {
        corrThresholdSlider.addEventListener('input', (e) => {
            currentCorrThreshold = parseFloat(e.target.value) || 0.0;
            if (corrThresholdDisplay) {
                corrThresholdDisplay.textContent = currentCorrThreshold.toFixed(2);
            }
            if (currentCorrelation) {
                renderRankedCorrelationsTable(currentCorrelation.all_ranked_pairs || currentCorrelation.ranked_pairs || []);
            }
        });
    }

    // Correlation Table Search Listener
    if (corrTableSearch) {
        corrTableSearch.addEventListener('input', (e) => {
            corrSearchQuery = e.target.value.toLowerCase().trim();
            if (currentCorrelation) {
                renderRankedCorrelationsTable(currentCorrelation.all_ranked_pairs || currentCorrelation.ranked_pairs || []);
            }
        });
    }

    // =========================================================
    // PHASE 7: OUTLIER & ANOMALY DETECTION CONTROLLER
    // =========================================================
    let currentOutliers = null;
    let currentOutlierMethod = 'iqr';
    let currentOutlierParam = 1.5;
    let currentOutlierVizMode = 'boxplot'; // 'boxplot' | 'distribution'
    let currentOutlierSelectedCol = '';
    let outlierTableSearchQuery = '';
    let currentBoxplotSpec = null;

    const outliersCountBadge = document.getElementById('outliers-count-badge');
    const outlierMethodSelect = document.getElementById('outlier-method-select');
    const outlierParamInput = document.getElementById('outlier-param-input');
    const outlierParamLabel = document.getElementById('outlier-param-label');
    const btnRecalcOutliers = document.getElementById('btn-recalc-outliers');

    const outliersKpiNumCount = document.getElementById('outliers-kpi-num-count');
    const outliersKpiAffectedCols = document.getElementById('outliers-kpi-affected-cols');
    const outliersKpiTotalCount = document.getElementById('outliers-kpi-total-count');
    const outliersKpiErrorCount = document.getElementById('outliers-kpi-error-count');
    const outliersKpiPotentialCount = document.getElementById('outliers-kpi-potential-count');
    const outliersKpiCleanCount = document.getElementById('outliers-kpi-clean-count');
    const outliersWarningsContainer = document.getElementById('outliers-warnings-container');

    const viewOutlierBoxplotBtn = document.getElementById('view-outlier-boxplot-btn');
    const viewOutlierDistBtn = document.getElementById('view-outlier-dist-btn');
    const outlierDistSelectWrap = document.getElementById('outlier-dist-select-wrap');
    const outlierDistColSelect = document.getElementById('outlier-dist-col-select');
    const outlierPlotlyChart = document.getElementById('outlier-plotly-chart');

    const outlierTableSearch = document.getElementById('outlier-table-search');
    const outliersSummaryTbody = document.getElementById('outliers-summary-tbody');

    const btnOutliersKeep = document.getElementById('btn-outliers-keep');
    const btnOutliersRemove = document.getElementById('btn-outliers-remove');
    const btnOutliersCap = document.getElementById('btn-outliers-cap');
    const btnOutliersRemoveErrors = document.getElementById('btn-outliers-remove-errors');

    const outlierInspectorModal = document.getElementById('outlier-inspector-modal');
    const modalOutlierTitle = document.getElementById('modal-outlier-title');
    const modalOutlierMeta = document.getElementById('modal-outlier-meta');
    const modalOutlierTbody = document.getElementById('modal-outlier-tbody');
    const modalOutlierCloseBtn = document.getElementById('modal-outlier-close-btn');

    async function loadOutlierData(datasetId, method = currentOutlierMethod, param = currentOutlierParam) {
        if (!datasetId) return;

        try {
            const url = `/api/outliers/${encodeURIComponent(datasetId)}?method=${encodeURIComponent(method)}&param=${encodeURIComponent(param)}`;
            const res = await fetch(url);
            const data = await res.json();

            if (!res.ok || !data.success) {
                console.warn('Outlier detection API returned non-success:', data);
                return;
            }

            currentOutliers = data.outliers;
            currentBoxplotSpec = data.boxplot_spec;

            // 1. Update KPI Summary
            renderOutlierKPIs(currentOutliers);

            // 2. Update Tab Badge Counter
            if (outliersCountBadge) {
                outliersCountBadge.textContent = (currentOutliers.total_outlier_instances || 0).toLocaleString();
            }

            // 3. Render Statistical Warnings Banner
            renderOutlierWarnings(currentOutliers.warnings || []);

            // 4. Populate Feature Distribution Selector Dropdown
            populateOutlierColumnSelector();

            // 5. Render Outlier Summary Table
            renderOutliersSummaryTable();

            // 6. Render Active Visualization Chart
            renderOutlierVisualization();

        } catch (err) {
            console.error('Failed to load outlier detection data:', err);
        }
    }

    function renderOutlierKPIs(outliers) {
        if (!outliers) return;

        if (outliersKpiNumCount) outliersKpiNumCount.textContent = (outliers.numerical_columns_count || 0).toLocaleString();
        if (outliersKpiAffectedCols) outliersKpiAffectedCols.textContent = (outliers.columns_with_outliers_count || 0).toLocaleString();
        if (outliersKpiTotalCount) outliersKpiTotalCount.textContent = (outliers.total_outlier_instances || 0).toLocaleString();
        if (outliersKpiErrorCount) outliersKpiErrorCount.textContent = (outliers.total_confirmed_errors || 0).toLocaleString();
        if (outliersKpiPotentialCount) outliersKpiPotentialCount.textContent = (outliers.total_potential_outliers || 0).toLocaleString();
        if (outliersKpiCleanCount) outliersKpiCleanCount.textContent = (outliers.clean_columns_count || 0).toLocaleString();
    }

    function renderOutlierWarnings(warnings) {
        if (!outliersWarningsContainer) return;

        if (!warnings || warnings.length === 0) {
            outliersWarningsContainer.innerHTML = '';
            outliersWarningsContainer.classList.add('hidden');
            return;
        }

        outliersWarningsContainer.classList.remove('hidden');
        outliersWarningsContainer.innerHTML = warnings.map((w) => {
            let cardClass = 'outlier-warning-card';
            let icon = '⚠️';

            if (w.toLowerCase().includes('zero standard deviation') || w.toLowerCase().includes('zero variance')) {
                cardClass += ' warning-zero-std';
                icon = '🛑';
            } else if (w.toLowerCase().includes('zero iqr')) {
                cardClass += ' warning-zero-iqr';
                icon = '📉';
            } else if (w.toLowerCase().includes('small sample size') || w.toLowerCase().includes('small dataset')) {
                icon = '🔬';
            }

            return `
                <div class="${cardClass}">
                    <span class="outlier-warning-icon">${icon}</span>
                    <div class="outlier-warning-text">
                        <strong>Statistical Edge Case:</strong> ${escapeHtml(w)}
                    </div>
                </div>
            `;
        }).join('');
    }

    function populateOutlierColumnSelector() {
        if (!outlierDistColSelect || !currentOutliers) return;

        const summaryTable = currentOutliers.summary_table || [];
        outlierDistColSelect.innerHTML = '<option value="">Select feature to inspect distribution...</option>';

        summaryTable.forEach((item) => {
            const opt = document.createElement('option');
            opt.value = item.column_name;
            const errStr = item.confirmed_error_count > 0 ? ` (${item.confirmed_error_count} errors)` : '';
            opt.textContent = `${item.column_name} [${item.outlier_count} outliers${errStr}]`;
            outlierDistColSelect.appendChild(opt);
        });

        if (!currentOutlierSelectedCol && summaryTable.length > 0) {
            // Default to first column with outliers, or first column overall
            const withOutliers = summaryTable.find((c) => c.outlier_count > 0);
            currentOutlierSelectedCol = withOutliers ? withOutliers.column_name : summaryTable[0].column_name;
            outlierDistColSelect.value = currentOutlierSelectedCol;
        } else if (currentOutlierSelectedCol) {
            outlierDistColSelect.value = currentOutlierSelectedCol;
        }
    }

    function renderOutliersSummaryTable() {
        if (!outliersSummaryTbody || !currentOutliers) return;

        let rows = currentOutliers.summary_table || [];

        if (outlierTableSearchQuery) {
            rows = rows.filter((r) => {
                const name = (r.column_name || '').toLowerCase();
                const status = (r.status || '').toLowerCase();
                return name.includes(outlierTableSearchQuery) || status.includes(outlierTableSearchQuery);
            });
        }

        if (rows.length === 0) {
            outliersSummaryTbody.innerHTML = `
                <tr>
                    <td colspan="11" style="text-align: center; color: var(--text-muted); padding: 30px;">
                        No numerical columns found matching search query '${escapeHtml(outlierTableSearchQuery)}'.
                    </td>
                </tr>
            `;
            return;
        }

        outliersSummaryTbody.innerHTML = rows.map((r, idx) => {
            const isClean = (r.outlier_count || 0) === 0;
            const hasErrors = (r.confirmed_error_count || 0) > 0;
            const outlierPct = r.outlier_percentage !== undefined ? Number(r.outlier_percentage).toFixed(2) : '0.00';
            const lowerVal = r.lower_threshold !== null && r.lower_threshold !== undefined ? Number(r.lower_threshold).toFixed(2) : '--';
            const upperVal = r.upper_threshold !== null && r.upper_threshold !== undefined ? Number(r.upper_threshold).toFixed(2) : '--';

            // Diagnosis badge
            let diagBadge = '';
            if (isClean) {
                diagBadge = '<span class="diag-badge diag-badge-clean">✓ Clean</span>';
            } else if (hasErrors) {
                diagBadge = `<span class="diag-badge diag-badge-error">⚠️ ${r.confirmed_error_count} Data Error${r.confirmed_error_count > 1 ? 's' : ''}</span>`;
            } else if (r.outlier_percentage > 5.0) {
                diagBadge = `<span class="diag-badge diag-badge-error">⚡ ${r.outlier_count} High Outliers</span>`;
            } else {
                diagBadge = `<span class="diag-badge diag-badge-potential">⚡ ${r.outlier_count} Outliers</span>`;
            }

            // Warnings badge
            let warnBadge = '<span style="color: var(--text-muted); font-size: 11px;">None</span>';
            if (r.warnings_count > 0) {
                const firstWarn = r.warnings && r.warnings[0] ? r.warnings[0] : 'Warning detected';
                warnBadge = `<span class="diag-badge diag-badge-warning" title="${escapeHtml(firstWarn)}">⚠️ ${r.warnings_count} Alert${r.warnings_count > 1 ? 's' : ''}</span>`;
            }

            // Action buttons
            const inspectDisabled = isClean ? 'disabled' : '';
            const actionButtons = `
                <div style="display: flex; gap: 6px;">
                    <button class="btn btn-outline btn-xs btn-inspect-outliers" data-col="${escapeHtml(r.column_name)}" ${inspectDisabled}>
                        🔍 Review (${r.outlier_count})
                    </button>
                    <button class="btn btn-outline btn-xs btn-view-outlier-chart" data-col="${escapeHtml(r.column_name)}">
                        📊 Chart
                    </button>
                </div>
            `;

            return `
                <tr>
                    <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;">${idx + 1}</td>
                    <td style="font-weight: 700; color: var(--text-primary);">${escapeHtml(r.column_name)}</td>
                    <td><code style="font-size: 11px; background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px;">${escapeHtml(r.method || 'IQR')}</code></td>
                    <td style="font-family: var(--font-mono);">${(r.valid_observations || 0).toLocaleString()}</td>
                    <td><strong style="font-family: var(--font-mono); color: ${isClean ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${(r.outlier_count || 0).toLocaleString()}</strong></td>
                    <td style="font-family: var(--font-mono); color: ${isClean ? 'var(--text-muted)' : 'var(--accent-rose)'};">${outlierPct}%</td>
                    <td style="font-family: var(--font-mono); color: var(--text-secondary);">${lowerVal}</td>
                    <td style="font-family: var(--font-mono); color: var(--text-secondary);">${upperVal}</td>
                    <td>${diagBadge}</td>
                    <td>${warnBadge}</td>
                    <td>${actionButtons}</td>
                </tr>
            `;
        }).join('');
    }

    async function renderOutlierVisualization() {
        if (!outlierPlotlyChart) return;

        if (!window.Plotly) {
            outlierPlotlyChart.innerHTML = `
                <div class="chart-loading-placeholder">
                    <p style="color: var(--accent-rose);">Plotly.js library unavailable.</p>
                </div>
            `;
            return;
        }

        if (currentOutlierVizMode === 'boxplot') {
            if (currentBoxplotSpec) {
                Plotly.react('outlier-plotly-chart', currentBoxplotSpec.data || [], currentBoxplotSpec.layout || {}, currentBoxplotSpec.config || { responsive: true });
            }
        } else {
            // Distribution mode for selected column
            if (!currentOutlierSelectedCol && currentOutliers && currentOutliers.summary_table && currentOutliers.summary_table.length > 0) {
                currentOutlierSelectedCol = currentOutliers.summary_table[0].column_name;
            }

            if (currentOutlierSelectedCol) {
                try {
                    const url = `/api/outliers/${encodeURIComponent(activeDatasetId)}/column/${encodeURIComponent(currentOutlierSelectedCol)}?method=${encodeURIComponent(currentOutlierMethod)}&param=${encodeURIComponent(currentOutlierParam)}`;
                    const res = await fetch(url);
                    const data = await res.json();
                    if (res.ok && data.success && data.distribution_spec) {
                        Plotly.react('outlier-plotly-chart', data.distribution_spec.data || [], data.distribution_spec.layout || {}, data.distribution_spec.config || { responsive: true });
                    }
                } catch (err) {
                    console.error('Failed to render distribution chart for column:', currentOutlierSelectedCol, err);
                }
            }
        }
    }

    function openOutlierInspectorModal(colName) {
        if (!outlierInspectorModal || !currentOutliers) return;

        const colDetails = (currentOutliers.column_details && currentOutliers.column_details[colName]) || null;
        if (!colDetails) {
            alert(`No outlier details available for column '${colName}'.`);
            return;
        }

        const outliers = colDetails.outliers || [];

        if (modalOutlierTitle) {
            modalOutlierTitle.textContent = `Outlier Records Drill-Down: ${colName}`;
        }
        if (modalOutlierMeta) {
            modalOutlierMeta.textContent = `${outliers.length} Flagged Observations | Lower: ${colDetails.lower_threshold} | Upper: ${colDetails.upper_threshold}`;
        }

        if (modalOutlierTbody) {
            if (outliers.length === 0) {
                modalOutlierTbody.innerHTML = `
                    <tr>
                        <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">
                            No outlier records found in column '${escapeHtml(colName)}'.
                        </td>
                    </tr>
                `;
            } else {
                modalOutlierTbody.innerHTML = outliers.map((o) => {
                    const isError = o.is_data_error;
                    const tagClass = isError ? 'class-tag-error' : 'class-tag-potential';
                    const boundVi = o.direction === 'below_lower' ? 'Below Lower Bound' : 'Above Upper Bound';
                    const threshVal = o.direction === 'below_lower' ? o.lower_threshold : o.upper_threshold;

                    return `
                        <tr>
                            <td style="font-family: var(--font-mono); font-weight: 700; color: var(--text-primary);">${o.row_index}</td>
                            <td><strong style="font-family: var(--font-mono); color: ${isError ? 'var(--accent-rose)' : 'var(--accent-amber)'}; font-size: 13px;">${o.value}</strong></td>
                            <td style="font-size: 12px; color: var(--text-secondary);">${boundVi}</td>
                            <td style="font-family: var(--font-mono); font-size: 12px;">${threshVal !== null ? Number(threshVal).toFixed(2) : '--'}</td>
                            <td style="font-family: var(--font-mono); font-size: 12px; color: var(--accent-rose);">+${o.deviation_from_threshold !== null ? Number(o.deviation_from_threshold).toFixed(2) : '--'}</td>
                            <td><span class="class-tag ${tagClass}">${escapeHtml(o.classification || (isError ? 'Confirmed Data Error' : 'Potential Outlier'))}</span></td>
                            <td style="font-size: 12px; color: var(--text-secondary); line-height: 1.4;">${escapeHtml(o.rationale || '')}</td>
                        </tr>
                    `;
                }).join('');
            }
        }

        outlierInspectorModal.classList.remove('hidden');
    }

    async function executeOutlierRemediation(action) {
        if (!activeDatasetId) {
            alert('Please select an active dataset first.');
            return;
        }

        let confirmMsg = '';
        if (action === 'remove') {
            confirmMsg = 'Are you sure you want to remove all rows containing outliers?\n\nThis creates a separate new dataset in data/processed/. Your original raw CSV will not be modified.';
        } else if (action === 'remove_errors_only') {
            confirmMsg = 'Are you sure you want to remove rows with confirmed data errors?\n\nLegitimate statistical outliers will be kept. Creates a new dataset in data/processed/.';
        } else if (action === 'cap') {
            confirmMsg = 'Are you sure you want to cap/winsorize extreme outliers to the boundary thresholds?\n\nNo rows will be dropped. Creates a new dataset in data/processed/.';
        }

        if (!confirm(confirmMsg)) return;

        try {
            const res = await fetch(`/api/outliers/remediate/${encodeURIComponent(activeDatasetId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: action,
                    method: currentOutlierMethod,
                    param: currentOutlierParam,
                }),
            });

            const data = await res.json();
            if (res.ok && data.success) {
                alert(`✓ Outlier remediation complete!\n\n${data.message}\nSaved as processed dataset: ${data.processed_id}`);
                // Refresh datasets list & reload dashboard
                await loadDatasetsList();
                if (activeDatasetStatus) {
                    activeDatasetStatus.classList.remove('hidden');
                    activeDatasetStatus.textContent = 'Cleaned & Processed Dataset Available';
                }
            } else {
                alert(`Error applying remediation: ${data.error || 'Operation failed.'}`);
            }
        } catch (err) {
            alert(`Remediation request failed: ${err.message}`);
        }
    }

    // Event: Outlier Method Selection
    if (outlierMethodSelect) {
        outlierMethodSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            if (val === 'iqr') {
                currentOutlierMethod = 'iqr';
                currentOutlierParam = 1.5;
                if (outlierParamLabel) outlierParamLabel.textContent = 'IQR Multiplier:';
            } else if (val === 'iqr_extreme') {
                currentOutlierMethod = 'iqr';
                currentOutlierParam = 3.0;
                if (outlierParamLabel) outlierParamLabel.textContent = 'IQR Multiplier:';
            } else if (val === 'zscore') {
                currentOutlierMethod = 'zscore';
                currentOutlierParam = 3.0;
                if (outlierParamLabel) outlierParamLabel.textContent = 'Z-Score (σ):';
            } else if (val === 'zscore_sensitive') {
                currentOutlierMethod = 'zscore';
                currentOutlierParam = 2.5;
                if (outlierParamLabel) outlierParamLabel.textContent = 'Z-Score (σ):';
            } else if (val === 'modified_zscore') {
                currentOutlierMethod = 'modified_zscore';
                currentOutlierParam = 3.5;
                if (outlierParamLabel) outlierParamLabel.textContent = 'MAD Threshold:';
            }

            if (outlierParamInput) outlierParamInput.value = currentOutlierParam;
            loadOutlierData(activeDatasetId, currentOutlierMethod, currentOutlierParam);
        });
    }

    // Event: Recalculate Outlier Bounds
    if (btnRecalcOutliers) {
        btnRecalcOutliers.addEventListener('click', () => {
            if (outlierParamInput) {
                currentOutlierParam = parseFloat(outlierParamInput.value) || 1.5;
            }
            loadOutlierData(activeDatasetId, currentOutlierMethod, currentOutlierParam);
        });
    }

    // Event: Toggle Visualizations
    if (viewOutlierBoxplotBtn) {
        viewOutlierBoxplotBtn.addEventListener('click', () => {
            currentOutlierVizMode = 'boxplot';
            viewOutlierBoxplotBtn.classList.add('active');
            if (viewOutlierDistBtn) viewOutlierDistBtn.classList.remove('active');
            if (outlierDistSelectWrap) outlierDistSelectWrap.classList.add('hidden');
            renderOutlierVisualization();
        });
    }

    if (viewOutlierDistBtn) {
        viewOutlierDistBtn.addEventListener('click', () => {
            currentOutlierVizMode = 'distribution';
            viewOutlierDistBtn.classList.add('active');
            if (viewOutlierBoxplotBtn) viewOutlierBoxplotBtn.classList.remove('active');
            if (outlierDistSelectWrap) outlierDistSelectWrap.classList.remove('hidden');
            renderOutlierVisualization();
        });
    }

    if (outlierDistColSelect) {
        outlierDistColSelect.addEventListener('change', (e) => {
            currentOutlierSelectedCol = e.target.value;
            renderOutlierVisualization();
        });
    }

    // Event: Outlier Table Search Filter
    if (outlierTableSearch) {
        outlierTableSearch.addEventListener('input', (e) => {
            outlierTableSearchQuery = e.target.value.toLowerCase().trim();
            renderOutliersSummaryTable();
        });
    }

    // Event Delegation: Outlier Table Actions (Inspect & Chart buttons)
    if (outliersSummaryTbody) {
        outliersSummaryTbody.addEventListener('click', (e) => {
            const inspectBtn = e.target.closest('.btn-inspect-outliers');
            if (inspectBtn) {
                const colName = inspectBtn.getAttribute('data-col');
                if (colName) openOutlierInspectorModal(colName);
                return;
            }

            const chartBtn = e.target.closest('.btn-view-outlier-chart');
            if (chartBtn) {
                const colName = chartBtn.getAttribute('data-col');
                if (colName) {
                    currentOutlierSelectedCol = colName;
                    currentOutlierVizMode = 'distribution';
                    if (viewOutlierDistBtn) viewOutlierDistBtn.classList.add('active');
                    if (viewOutlierBoxplotBtn) viewOutlierBoxplotBtn.classList.remove('active');
                    if (outlierDistSelectWrap) outlierDistSelectWrap.classList.remove('hidden');
                    if (outlierDistColSelect) outlierDistColSelect.value = colName;
                    renderOutlierVisualization();

                    // Smooth scroll to chart
                    const chartCard = document.querySelector('.outliers-viz-section');
                    if (chartCard) chartCard.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    }

    // Modal Close Events
    if (modalOutlierCloseBtn && outlierInspectorModal) {
        modalOutlierCloseBtn.addEventListener('click', () => {
            outlierInspectorModal.classList.add('hidden');
        });
        outlierInspectorModal.addEventListener('click', (e) => {
            if (e.target === outlierInspectorModal) {
                outlierInspectorModal.classList.add('hidden');
            }
        });
    }

    // Remediation Buttons
    if (btnOutliersKeep) {
        btnOutliersKeep.addEventListener('click', () => {
            alert('✓ Outliers retained. All extreme observations marked as valid domain data.');
        });
    }

    if (btnOutliersRemoveErrors) {
        btnOutliersRemoveErrors.addEventListener('click', () => executeOutlierRemediation('remove_errors_only'));
    }

    if (btnOutliersCap) {
        btnOutliersCap.addEventListener('click', () => executeOutlierRemediation('cap'));
    }

    if (btnOutliersRemove) {
        btnOutliersRemove.addEventListener('click', () => executeOutlierRemediation('remove'));
    }

    // =========================================================================
    // PHASE 8: AUTOMATIC VISUALIZATION ENGINE & CUSTOM CHART BUILDER
    // =========================================================================

    async function loadVisualizationData(datasetId) {
        if (!datasetId) return;
        try {
            const res = await fetch(`/api/visualization/recommendations/${encodeURIComponent(datasetId)}?limit=12`);
            const data = await res.json();
            if (res.ok && data.success) {
                allRecommendations = data.recommendations || [];
                vizSchema = data.schema || {};

                // Update badge counter
                if (vizCountBadge) vizCountBadge.textContent = allRecommendations.length;
                if (recFilterAllCount) recFilterAllCount.textContent = allRecommendations.length;

                // Compute breakdown KPIs
                let distCount = 0, relCount = 0, catCount = 0, trendCount = 0;
                allRecommendations.forEach((r) => {
                    const cat = (r.category || '').toLowerCase();
                    if (cat.includes('dist')) distCount++;
                    else if (cat.includes('rel')) relCount++;
                    else if (cat.includes('cat')) catCount++;
                    else if (cat.includes('trend')) trendCount++;
                });

                if (vizKpiTotalCount) vizKpiTotalCount.textContent = allRecommendations.length;
                if (vizKpiDistCount) vizKpiDistCount.textContent = distCount;
                if (vizKpiRelCount) vizKpiRelCount.textContent = relCount;
                if (vizKpiCatCount) vizKpiCatCount.textContent = catCount;
                if (vizKpiTrendCount) vizKpiTrendCount.textContent = trendCount;

                renderRecommendedCharts();
                populateCustomChartControls(allColumnsProfile, vizSchema);
            }
        } catch (err) {
            console.error('Failed to load visualization data:', err);
        }
    }

    function renderRecommendedCharts() {
        if (!recommendedChartsGrid) return;

        const filtered = activeVizFilter === 'all'
            ? allRecommendations
            : allRecommendations.filter((r) => (r.category || '').toLowerCase() === activeVizFilter.toLowerCase());

        if (filtered.length === 0) {
            recommendedChartsGrid.innerHTML = `
                <div class="viz-empty-card">
                    <span class="empty-icon">📊</span>
                    <h4>No Recommended Visualizations in this Category</h4>
                    <p>Try switching filter tabs or use "Build Your Own Chart" below to create custom visualizations.</p>
                </div>
            `;
            return;
        }

        recommendedChartsGrid.innerHTML = '';

        filtered.forEach((rec, idx) => {
            const card = document.createElement('div');
            card.className = 'rec-chart-card';

            const priorityBadgeClass = rec.priority === 'High' ? 'badge-rose' : 'badge-cyan';
            const chartId = `plotly-rec-${rec.id}`;

            card.innerHTML = `
                <div class="rec-chart-header">
                    <div class="rec-title-wrap">
                        <div class="rec-badges-row">
                            <span class="rec-rank-badge">#${idx + 1}</span>
                            <span class="modal-type-badge ${priorityBadgeClass}">${escapeHtml(rec.priority)} Priority</span>
                            <span class="rec-type-badge">${escapeHtml((rec.chart_type || '').toUpperCase())}</span>
                            <span class="rec-category-badge">${escapeHtml(rec.category || '')}</span>
                        </div>
                        <h4 class="rec-chart-title">${escapeHtml(rec.title)}</h4>
                    </div>
                </div>
                
                <div class="rec-rationale-box">
                    <span class="rationale-icon">💡</span>
                    <p class="rationale-text">${escapeHtml(rec.rationale)}</p>
                </div>

                <div class="rec-plotly-container" id="${chartId}">
                    <div class="chart-loading-placeholder">
                        <div class="spinner"></div>
                    </div>
                </div>

                <div class="rec-chart-footer">
                    <div class="rec-cols-used">
                        <span class="cols-label">Features:</span>
                        ${(rec.columns_used || []).map(c => `<code class="col-pill">${escapeHtml(c)}</code>`).join(' ')}
                    </div>
                </div>
            `;

            recommendedChartsGrid.appendChild(card);

            if (window.Plotly && rec.plotly_spec) {
                setTimeout(() => {
                    const el = document.getElementById(chartId);
                    if (el) {
                        Plotly.newPlot(chartId, rec.plotly_spec.data || [], rec.plotly_spec.layout || {}, rec.plotly_spec.config || PLOTLY_CONFIG);
                    }
                }, 40);
            }
        });
    }

    function populateCustomChartControls(columns, schema) {
        if (!builderXCol || !builderYCol || !builderColorCol) return;

        const colNames = (columns || []).map(c => (typeof c === 'object' && c !== null ? c.name : c)).filter(Boolean);

        const prevX = builderXCol.value;
        const prevY = builderYCol.value;
        const prevColor = builderColorCol.value;

        builderXCol.innerHTML = '<option value="">Select X Column...</option>';
        builderYCol.innerHTML = '<option value="">None / Frequency Count</option>';
        builderColorCol.innerHTML = '<option value="">No Grouping</option>';

        colNames.forEach((col) => {
            const optX = document.createElement('option');
            optX.value = col;
            optX.textContent = col;
            if (col === prevX) optX.selected = true;
            builderXCol.appendChild(optX);

            const optY = document.createElement('option');
            optY.value = col;
            optY.textContent = col;
            if (col === prevY) optY.selected = true;
            builderYCol.appendChild(optY);

            const optColor = document.createElement('option');
            optColor.value = col;
            optColor.textContent = col;
            if (col === prevColor) optColor.selected = true;
            builderColorCol.appendChild(optColor);
        });

        if (!builderXCol.value && colNames.length > 0) {
            builderXCol.value = colNames[0];
            if (colNames.length > 1) {
                builderYCol.value = colNames[1];
            }
        }
    }

    async function executeGenerateCustomChart() {
        if (!activeDatasetId) return;

        const xCol = builderXCol ? builderXCol.value : '';
        if (!xCol) {
            alert('Please select an X-Axis column.');
            return;
        }

        const yCol = builderYCol ? builderYCol.value : '';
        const chartType = builderChartType ? builderChartType.value : 'bar';
        const aggregation = builderAggregation ? builderAggregation.value : 'none';
        const colorCol = builderColorCol ? builderColorCol.value : '';
        const title = builderChartTitle ? builderChartTitle.value.trim() : '';

        if (customPlotlyCanvas) {
            customPlotlyCanvas.innerHTML = `
                <div class="chart-loading-placeholder">
                    <div class="spinner"></div>
                    <p>Generating custom ${escapeHtml(chartType)} chart specification...</p>
                </div>
            `;
        }

        try {
            const res = await fetch(`/api/visualization/custom/${encodeURIComponent(activeDatasetId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chart_type: chartType,
                    x_col: xCol,
                    y_col: yCol || null,
                    color_col: colorCol || null,
                    aggregation: aggregation,
                    title: title || null,
                }),
            });

            const data = await res.json();
            if (res.ok && data.success && data.spec && customPlotlyCanvas) {
                customPlotlyCanvas.innerHTML = '';
                Plotly.newPlot(
                    'custom-plotly-canvas',
                    data.spec.data || [],
                    data.spec.layout || {},
                    data.spec.config || PLOTLY_CONFIG
                );
            } else {
                if (customPlotlyCanvas) {
                    customPlotlyCanvas.innerHTML = `
                        <div class="builder-empty-placeholder text-rose">
                            <span class="empty-icon">⚠️</span>
                            <h4>Chart Generation Error</h4>
                            <p>${escapeHtml(data.error || 'Could not generate custom chart.')}</p>
                        </div>
                    `;
                }
            }
        } catch (err) {
            console.error('Failed to generate custom chart:', err);
            if (customPlotlyCanvas) {
                customPlotlyCanvas.innerHTML = `
                    <div class="builder-empty-placeholder text-rose">
                        <span class="empty-icon">⚠️</span>
                        <h4>Chart Generation Request Failed</h4>
                        <p>${escapeHtml(err.message)}</p>
                    </div>
                `;
            }
        }
    }

    // Event: Visualization Filter Buttons
    vizFilterBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            const filterVal = btn.getAttribute('data-filter') || 'all';
            activeVizFilter = filterVal;
            vizFilterBtns.forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            renderRecommendedCharts();
        });
    });

    // Event: Generate Custom Chart Button
    if (btnGenerateCustomChart) {
        btnGenerateCustomChart.addEventListener('click', () => {
            executeGenerateCustomChart();
        });
    }

    // Event: Reset Custom Chart Button
    if (btnResetCustomChart) {
        btnResetCustomChart.addEventListener('click', () => {
            if (builderChartType) builderChartType.value = 'bar';
            if (builderAggregation) builderAggregation.value = 'none';
            if (builderColorCol) builderColorCol.value = '';
            if (builderChartTitle) builderChartTitle.value = '';
            if (allColumnsProfile && allColumnsProfile.length > 0) {
                if (builderXCol) builderXCol.value = allColumnsProfile[0].name || '';
                if (builderYCol && allColumnsProfile.length > 1) {
                    builderYCol.value = allColumnsProfile[1].name || '';
                }
            }
            if (customPlotlyCanvas) {
                customPlotlyCanvas.innerHTML = `
                    <div class="builder-empty-placeholder">
                        <span class="empty-icon">📊</span>
                        <h4>Custom Chart Workbench Ready</h4>
                        <p>Select your X and Y columns above and click "Generate / Update Chart" to render an interactive visualization.</p>
                    </div>
                `;
            }
        });
    }

    // Initial Execution
    (async () => {
        await loadDatasetsList();
        await loadDatasetDashboard(activeDatasetId);
    })();
});



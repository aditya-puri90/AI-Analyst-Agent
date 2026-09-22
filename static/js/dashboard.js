/**
 * AI Data Analyst Agent Studio - Master Dashboard Controller (Phase 12)
 * Manages all 11 studio navigation views, executive dashboard widgets,
 * interactive Plotly visualizations, data quality auditing, non-destructive cleaning,
 * descriptive statistics, correlations, outlier remediation, AI insights,
 * conversational Q&A assistant, and executive report generation.
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // Application State
    // -------------------------------------------------------------
    let activeDatasetId = window.INITIAL_DATASET_ID || '';
    let currentProfile = null;
    let allColumnsProfile = [];
    let activeTypeFilter = 'all';
    let columnSearchQuery = '';

    // Quality & Cleaning State
    let allDetectedIssues = [];
    let activeSeverityFilter = 'all';
    let currentCleaningSummary = null;
    let isViewingCleanedData = false;

    // Explorer Table State
    let currentPage = 1;
    const pageSize = 20;
    let currentPreviewRows = [];
    let currentPreviewCols = [];

    // Statistical Engine State
    let currentStatistics = null;
    let statsSearchQuery = '';

    // Correlation Engine State
    let currentCorrelation = null;
    let currentHeatmapSpec = null;
    let currentCorrThreshold = 0.0;

    // Outlier Engine State
    let currentOutliers = null;
    let activeOutlierMethod = 'iqr';

    // Visualization Engine State
    let allRecommendations = [];
    let activeVizFilter = 'all';
    let vizSchema = null;

    // AI Insights State
    let currentInsights = null;

    // Report State
    let currentReport = null;

    // Chat State
    let isChatStreaming = false;

    // -------------------------------------------------------------
    // DOM Elements - Navigation & Global
    // -------------------------------------------------------------
    const datasetSelector = document.getElementById('dataset-selector');
    const loadingState = document.getElementById('dashboard-loading');
    const errorState = document.getElementById('dashboard-error');
    const contentState = document.getElementById('dashboard-content');
    const studioTabBtns = document.querySelectorAll('.studio-tab-btn');
    const studioTabContents = document.querySelectorAll('.studio-tab-content');

    // Header Meta
    const headerDatasetName = document.getElementById('header-dataset-name');
    const headerDatasetStatus = document.getElementById('header-dataset-status');
    const headerQuickStats = document.getElementById('header-quick-stats');
    const headerStatRows = document.getElementById('header-stat-rows');
    const headerStatCols = document.getElementById('header-stat-cols');
    const headerStatGrade = document.getElementById('header-stat-grade');

    // Sample Loaders
    const btnSampleEcommerce = document.getElementById('btn-sample-ecommerce');
    const btnSampleEmployee = document.getElementById('btn-sample-employee');
    const btnLoadSampleErr = document.getElementById('btn-load-sample-err');

    // -------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------
    function formatBytes(bytes, decimals = 2) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function formatNumber(num) {
        if (num === null || num === undefined || isNaN(num)) return '--';
        return Number(num).toLocaleString();
    }

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function showToast(message, icon = 'ℹ️', duration = 3500) {
        const toast = document.getElementById('toast-notification');
        const msgEl = document.getElementById('toast-message');
        const iconEl = document.getElementById('toast-icon');
        if (!toast || !msgEl) return;
        msgEl.textContent = message;
        if (iconEl) iconEl.textContent = icon;
        toast.classList.remove('hidden');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, duration);
    }

    // -------------------------------------------------------------
    // View Navigation & Tab Routing (11 Sections)
    // -------------------------------------------------------------
    function switchTab(targetTabId, updateHash = true) {
        studioTabBtns.forEach(btn => {
            const isMatch = btn.getAttribute('data-tab') === targetTabId;
            btn.classList.toggle('active', isMatch);
            btn.setAttribute('aria-selected', isMatch ? 'true' : 'false');
        });

        studioTabContents.forEach(content => {
            const isMatch = content.id === targetTabId;
            content.classList.toggle('active', isMatch);
            content.classList.toggle('hidden', !isMatch);
        });

        if (updateHash) {
            const hashName = targetTabId.replace('-view', '');
            window.location.hash = hashName;
        }

        // Trigger Plotly Resize on all visible chart containers
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
            const plotDivs = document.querySelectorAll('.js-plotly-plot');
            plotDivs.forEach(p => {
                try { Plotly.Plots.resize(p); } catch (e) {}
            });
        }, 80);

        // Lazy-load specific view engines if not yet loaded
        if (targetTabId === 'reports-view' && !currentReport) {
            loadExecutiveReport();
        } else if (targetTabId === 'chat-view') {
            loadChatSuggestions();
        }
    }

    studioTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            switchTab(target);
        });
    });

    // Jump Buttons inside Dashboard
    document.getElementById('dash-btn-view-quality')?.addEventListener('click', () => switchTab('quality-view'));
    document.getElementById('dash-btn-clean-missing')?.addEventListener('click', () => switchTab('cleaning-view'));
    document.getElementById('dash-btn-deduplicate')?.addEventListener('click', () => switchTab('cleaning-view'));
    document.getElementById('dash-btn-view-stats')?.addEventListener('click', () => switchTab('statistics-view'));
    document.getElementById('dash-btn-view-corr')?.addEventListener('click', () => switchTab('correlation-view'));
    document.getElementById('dash-btn-view-outliers')?.addEventListener('click', () => switchTab('outliers-view'));
    document.getElementById('dash-btn-open-viz-studio')?.addEventListener('click', () => switchTab('visualization-view'));
    document.getElementById('dash-btn-view-full-ai')?.addEventListener('click', () => switchTab('insights-view'));
    document.getElementById('dash-btn-jump-reports')?.addEventListener('click', () => switchTab('reports-view'));
    document.getElementById('dash-btn-jump-chat')?.addEventListener('click', () => switchTab('chat-view'));
    document.getElementById('btn-goto-cleaning')?.addEventListener('click', () => switchTab('cleaning-view'));

    // Handle Hash on Load
    function handleInitialHash() {
        const hash = window.location.hash.replace('#', '').trim();
        if (hash) {
            const matchingTab = `${hash}-view`;
            const tabEl = document.getElementById(matchingTab);
            if (tabEl) {
                switchTab(matchingTab, false);
            }
        }
    }

    // -------------------------------------------------------------
    // Dataset Loading & Pipeline Orchestration
    // -------------------------------------------------------------
    async function loadDatasetList() {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();
            if (data.success && data.datasets) {
                datasetSelector.innerHTML = '<option value="">Select an uploaded dataset...</option>';
                data.datasets.forEach(d => {
                    const opt = document.createElement('option');
                    const dsId = d.id || d.dataset_id;
                    const name = d.original_name || d.original_filename || d.filename || dsId;
                    const rows = d.rows_estimate || d.total_rows || d.rows || 0;
                    const cols = d.columns_count || d.total_columns || d.columns || 0;
                    opt.value = dsId;
                    opt.textContent = `${name} (${rows} rows, ${cols} cols)`;
                    if (dsId === activeDatasetId) {
                        opt.selected = true;
                    }
                    datasetSelector.appendChild(opt);
                });
            }
        } catch (err) {
            console.error('Failed to load dataset list:', err);
        }
    }

    async function loadFullDatasetAnalytics(datasetId) {
        if (!datasetId || datasetId === 'undefined') {
            loadingState.classList.add('hidden');
            contentState.classList.add('hidden');
            errorState.classList.remove('hidden');
            return;
        }

        activeDatasetId = datasetId;
        try {
            localStorage.setItem('active_dataset_id', datasetId);
        } catch (e) {}

        if (datasetSelector) {
            datasetSelector.value = datasetId;
        }
        loadingState.classList.remove('hidden');
        errorState.classList.add('hidden');
        contentState.classList.add('hidden');

        try {
            // Update URL without reload
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('dataset_id', datasetId);
            window.history.replaceState({}, '', newUrl);

            // 1. Fetch Profile & Overview (Primary requirement)
            const profRes = await fetch(`/api/profile/${datasetId}`);
            const profData = await profRes.json();
            if (!profData.success) throw new Error(profData.error || 'Failed to profile dataset');
            currentProfile = profData.profile;
            allColumnsProfile = currentProfile.columns || [];

            // 2. Fetch Data Quality Audit (Fault-tolerant)
            let qualData = { success: false, issues: [] };
            try {
                const qualRes = await fetch(`/api/cleaning/audit/${datasetId}`);
                qualData = await qualRes.json();
            } catch (e) {
                console.warn('Quality audit fetch warning:', e);
            }
            allDetectedIssues = qualData.success ? qualData.issues : [];

            // 3. Fetch Descriptive Statistics (Fault-tolerant)
            let statsData = { success: false, statistics: null };
            try {
                const statsRes = await fetch(`/api/statistics/${datasetId}`);
                statsData = await statsRes.json();
            } catch (e) {
                console.warn('Statistics fetch warning:', e);
            }
            currentStatistics = statsData.success ? statsData.statistics : null;

            // 4. Fetch Correlation Analysis & Heatmap (Fault-tolerant)
            let corrData = { success: false, correlation: null, heatmap_spec: null };
            try {
                const corrRes = await fetch(`/api/correlation/${datasetId}`);
                corrData = await corrRes.json();
            } catch (e) {
                console.warn('Correlation fetch warning:', e);
            }
            currentCorrelation = corrData.success ? corrData.correlation : null;
            currentHeatmapSpec = corrData.success ? corrData.heatmap_spec : null;

            // 5. Fetch Outlier Detection (Fault-tolerant)
            let outData = { success: false, outliers: null, boxplot_spec: null };
            try {
                const outRes = await fetch(`/api/outliers/${datasetId}?method=${activeOutlierMethod}`);
                outData = await outRes.json();
            } catch (e) {
                console.warn('Outliers fetch warning:', e);
            }
            currentOutliers = outData.success ? outData.outliers : null;

            // 6. Fetch Chart Recommendations & Schema (Fault-tolerant)
            let vizData = { success: false, recommendations: [], schema: null };
            try {
                const vizRes = await fetch(`/api/visualization/recommendations/${datasetId}`);
                vizData = await vizRes.json();
            } catch (e) {
                console.warn('Visualizations fetch warning:', e);
            }
            allRecommendations = vizData.success ? vizData.recommendations : [];
            vizSchema = vizData.success ? vizData.schema : null;

            // 7. Fetch AI Insights (Fault-tolerant)
            let aiData = { success: false, insights: null };
            try {
                const aiRes = await fetch(`/api/insights/${datasetId}`);
                aiData = await aiRes.json();
            } catch (e) {
                console.warn('AI Insights fetch warning:', e);
            }
            currentInsights = aiData.success ? aiData.insights : null;

            // 8. Render All Views safely
            try { renderHeaderMeta(); } catch (e) { console.error('Error in renderHeaderMeta:', e); }
            try { renderDashboardExecutiveWidgets(); } catch (e) { console.error('Error in renderDashboardExecutiveWidgets:', e); }
            try { renderDataPreviewExplorer(); } catch (e) { console.error('Error in renderDataPreviewExplorer:', e); }
            try { renderDataQualityAudit(); } catch (e) { console.error('Error in renderDataQualityAudit:', e); }
            try { renderCleaningStudio(qualData); } catch (e) { console.error('Error in renderCleaningStudio:', e); }
            try { renderStatisticsView(); } catch (e) { console.error('Error in renderStatisticsView:', e); }
            try { renderCorrelationView(); } catch (e) { console.error('Error in renderCorrelationView:', e); }
            try { renderOutliersView(outData.boxplot_spec); } catch (e) { console.error('Error in renderOutliersView:', e); }
            try { renderVisualizationsView(); } catch (e) { console.error('Error in renderVisualizationsView:', e); }
            try { renderAiInsightsView(); } catch (e) { console.error('Error in renderAiInsightsView:', e); }

            // Reveal UI
            loadingState.classList.add('hidden');
            contentState.classList.remove('hidden');

            handleInitialHash();
        } catch (err) {
            console.error('Error loading dataset analytics:', err);
            loadingState.classList.add('hidden');
            contentState.classList.add('hidden');
            errorState.classList.remove('hidden');
            const errTitle = document.getElementById('error-title');
            const errMsg = document.getElementById('error-message');
            if (errTitle) errTitle.textContent = 'Analysis Ingestion Error';
            if (errMsg) errMsg.textContent = err.message || 'Could not load dataset analytics.';
        }
    }

    // -------------------------------------------------------------
    // Header & Meta Bar Rendering
    // -------------------------------------------------------------
    function renderHeaderMeta() {
        if (!currentProfile) return;
        const ov = currentProfile.overview || {};
        const qual = currentProfile.quality || currentProfile.quality_summary || {};
        const filename = currentProfile.original_filename || currentProfile.file_name || activeDatasetId;

        if (headerDatasetName) headerDatasetName.textContent = filename;
        if (headerDatasetStatus) {
            headerDatasetStatus.textContent = currentProfile.is_processed ? 'Cleaned' : 'Raw';
            headerDatasetStatus.className = currentProfile.is_processed ? 'meta-badge badge-emerald' : 'meta-badge';
        }

        if (headerQuickStats) headerQuickStats.classList.remove('hidden');
        if (headerStatRows) headerStatRows.textContent = `${formatNumber(ov.total_rows)} rows`;
        if (headerStatCols) headerStatCols.textContent = `${formatNumber(ov.total_columns)} cols`;
        if (headerStatGrade) headerStatGrade.textContent = `Grade: ${qual.health_grade || 'A+'}`;

        // Top banner title
        const dashTitle = document.getElementById('dash-active-dataset-name');
        if (dashTitle) dashTitle.textContent = `${filename}`;

        const dashBadge = document.getElementById('dash-processed-badge');
        if (dashBadge) {
            dashBadge.classList.toggle('hidden', !currentProfile.is_processed);
        }
    }

    // -------------------------------------------------------------
    // VIEW 1: EXECUTIVE DASHBOARD (10 MANDATORY WIDGETS)
    // -------------------------------------------------------------
    function renderDashboardExecutiveWidgets() {
        if (!currentProfile) return;
        const ov = currentProfile.overview || {};
        const qual = currentProfile.quality || currentProfile.quality_summary || {};
        const cols = currentProfile.columns || [];

        // Widget 1: Dataset Overview Cards
        const kpiRows = document.getElementById('kpi-rows');
        const kpiCols = document.getElementById('kpi-cols');
        const kpiCells = document.getElementById('kpi-cells');
        const kpiMem = document.getElementById('kpi-memory');
        const kpiFileSize = document.getElementById('kpi-file-size');
        const kpiProcStatus = document.getElementById('kpi-processing-status');

        if (kpiRows) kpiRows.textContent = formatNumber(ov.total_rows);
        if (kpiCols) kpiCols.textContent = formatNumber(ov.total_columns);
        if (kpiCells) kpiCells.textContent = formatNumber(ov.total_cells);
        if (kpiMem) kpiMem.textContent = `${Number(ov.memory_usage_mb || 0).toFixed(2)} MB`;
        if (kpiFileSize) kpiFileSize.textContent = formatBytes(ov.file_size_bytes || (ov.memory_usage_mb * 1024 * 1024));
        if (kpiProcStatus) {
            kpiProcStatus.textContent = currentProfile.is_processed ? 'Cleaned (Processed)' : 'Raw Upload';
            kpiProcStatus.className = currentProfile.is_processed ? 'kpi-value text-emerald' : 'kpi-value';
        }

        // Widget 2: Data Quality Score
        const score = Math.round(qual.health_score !== undefined ? qual.health_score : (qual.overall_score !== undefined ? qual.overall_score : 100));
        const grade = qual.health_grade || 'A+';
        const qualityScoreEl = document.getElementById('dash-quality-score');
        const qualityGradeEl = document.getElementById('dash-quality-grade');
        const qualityStatusEl = document.getElementById('dash-quality-status');
        const gaugeFill = document.getElementById('dash-gauge-fill');
        const rulesPassedEl = document.getElementById('dash-rules-passed');
        const issuesDetectedEl = document.getElementById('dash-issues-detected');

        if (qualityScoreEl) qualityScoreEl.textContent = score;
        if (qualityGradeEl) qualityGradeEl.textContent = grade;
        
        // Gauge stroke animation (circumference 2 * PI * 50 = 314)
        if (gaugeFill) {
            const offset = 314 - (314 * (score / 100));
            gaugeFill.style.strokeDashoffset = offset;
            gaugeFill.style.stroke = score >= 85 ? '#10b981' : score >= 70 ? '#f59e0b' : '#ef4444';
        }

        const totalIssuesCount = allDetectedIssues.length;
        const totalRulesCount = 12;
        const passedRules = Math.max(0, totalRulesCount - totalIssuesCount);

        if (rulesPassedEl) rulesPassedEl.textContent = `${passedRules} / ${totalRulesCount}`;
        if (issuesDetectedEl) {
            issuesDetectedEl.textContent = `${totalIssuesCount} defect${totalIssuesCount === 1 ? '' : 's'}`;
            issuesDetectedEl.className = totalIssuesCount === 0 ? 'text-emerald' : 'text-amber';
        }
        if (qualityStatusEl) {
            qualityStatusEl.textContent = qual.quality_status || (score >= 90 ? 'Optimal Data Health' : score >= 75 ? 'Good (Minor Defects)' : 'Action Required');
            qualityStatusEl.style.color = score >= 90 ? '#34d399' : score >= 75 ? '#fbbf24' : '#fb7185';
        }

        // Update badge counter on navigation tab
        const qBadge = document.getElementById('quality-issues-count-badge');
        if (qBadge) qBadge.textContent = totalIssuesCount;

        // Widget 3: Missing Value Summary
        const missingCells = ov.total_missing_cells !== undefined ? ov.total_missing_cells : (ov.missing_cells || 0);
        const missingPct = Number(ov.missing_cells_percentage || 0);
        const completeness = Math.max(0, 100 - missingPct);

        const missCountEl = document.getElementById('dash-missing-cells-count');
        const completenessEl = document.getElementById('dash-completeness-rate');
        const missBadge = document.getElementById('dash-missing-badge');
        const missColsAffectedEl = document.getElementById('dash-missing-cols-affected');
        const topMissingBars = document.getElementById('dash-top-missing-bars');

        if (missCountEl) missCountEl.textContent = formatNumber(missingCells);
        if (completenessEl) completenessEl.textContent = `${completeness.toFixed(1)}%`;
        if (missBadge) missBadge.textContent = `${missingPct.toFixed(1)}% Missing`;

        const colsWithMissing = cols.filter(c => (c.missing_count || 0) > 0);
        if (missColsAffectedEl) missColsAffectedEl.textContent = `${colsWithMissing.length} of ${cols.length}`;

        if (topMissingBars) {
            if (colsWithMissing.length === 0) {
                topMissingBars.innerHTML = '<div class="empty-state-text">✓ Zero missing values detected. Perfect 100% data completeness!</div>';
            } else {
                colsWithMissing.sort((a, b) => (b.missing_percentage || 0) - (a.missing_percentage || 0));
                topMissingBars.innerHTML = colsWithMissing.slice(0, 4).map(c => {
                    const colName = c.column_name || c.name;
                    return `
                    <div class="missing-bar-row">
                        <div class="missing-bar-labels">
                            <span class="missing-bar-col-name">${escapeHtml(colName)}</span>
                            <span class="missing-bar-pct">${c.missing_count} nulls (${Number(c.missing_percentage || 0).toFixed(1)}%)</span>
                        </div>
                        <div class="missing-bar-bg">
                            <div class="missing-bar-fill" style="width: ${Math.min(100, Math.max(4, c.missing_percentage))}%;"></div>
                        </div>
                    </div>
                `;
                }).join('');
            }
        }

        // Widget 4: Duplicate Summary
        const dupRows = ov.duplicate_rows || 0;
        const dupPct = Number(ov.duplicate_rows_percentage || 0);
        const uniqueness = Math.max(0, 100 - dupPct);

        const dupCountEl = document.getElementById('dash-dup-rows-count');
        const dupPctEl = document.getElementById('dash-dup-pct');
        const uniqEl = document.getElementById('dash-uniqueness-score');
        const dupBadge = document.getElementById('dash-dup-badge');
        const dupDesc = document.getElementById('dash-dup-desc');

        if (dupCountEl) dupCountEl.textContent = formatNumber(dupRows);
        if (dupPctEl) dupPctEl.textContent = `${dupPct.toFixed(1)}%`;
        if (uniqEl) uniqEl.textContent = `${uniqueness.toFixed(1)}%`;
        if (dupBadge) {
            dupBadge.textContent = dupRows === 0 ? '0 Duplicates' : `${formatNumber(dupRows)} Duplicates`;
            dupBadge.className = dupRows === 0 ? 'stat-tag-emerald' : 'stat-tag-amber';
        }
        if (dupDesc) {
            dupDesc.textContent = dupRows === 0
                ? 'Zero duplicate records identified. Every observation in the dataset is unique.'
                : `Detected ${formatNumber(dupRows)} duplicate record${dupRows === 1 ? '' : 's'} (${dupPct.toFixed(2)}% redundancy rate). Recommend deduplication.`;
        }

        // Widget 5: Numerical / Categorical Column Counts
        let numCount = 0, catCount = 0, dateCount = 0, boolCount = 0, otherCount = 0;
        cols.forEach(c => {
            const t = (c.classified_type || '').toLowerCase();
            if (t === 'numerical') numCount++;
            else if (t === 'categorical') catCount++;
            else if (t === 'datetime') dateCount++;
            else if (t === 'boolean') boolCount++;
            else otherCount++;
        });

        const totalCols = cols.length || 1;
        const numPct = (numCount / totalCols) * 100;
        const catPct = (catCount / totalCols) * 100;
        const datePct = (dateCount / totalCols) * 100;
        const boolPct = (boolCount / totalCols) * 100;

        const countNum = document.getElementById('dash-count-num');
        const countCat = document.getElementById('dash-count-cat');
        const countDate = document.getElementById('dash-count-date');
        const countBool = document.getElementById('dash-count-bool');
        const featPill = document.getElementById('dash-total-features-pill');
        if (countNum) countNum.textContent = numCount;
        if (countCat) countCat.textContent = catCount;
        if (countDate) countDate.textContent = dateCount;
        if (countBool) countBool.textContent = boolCount;
        if (featPill) featPill.textContent = `${totalCols} Features`;

        const distBar = document.getElementById('dash-type-dist-bar');
        if (distBar) {
            distBar.innerHTML = `
                <div class="type-bar-segment seg-num" style="width: ${numPct}%;" title="Numerical: ${numCount}"></div>
                <div class="type-bar-segment seg-cat" style="width: ${catPct}%;" title="Categorical: ${catCount}"></div>
                <div class="type-bar-segment seg-date" style="width: ${datePct}%;" title="Datetime: ${dateCount}"></div>
                <div class="type-bar-segment seg-bool" style="width: ${boolPct}%;" title="Boolean: ${boolCount}"></div>
            `;
        }

        // Widget 6: Important Statistical Findings
        const statFindingsList = document.getElementById('dash-stat-findings-list');
        if (statFindingsList && currentStatistics) {
            const obs = currentStatistics.statistical_observations || [];
            if (obs.length === 0) {
                statFindingsList.innerHTML = `
                    <div class="finding-item-box">
                        <span class="finding-badge badge-cyan">Uniform</span>
                        <p class="finding-text">Standard bell-curve and nominal distributions observed across features.</p>
                    </div>
                `;
            } else {
                statFindingsList.innerHTML = obs.slice(0, 4).map(o => `
                    <div class="finding-item-box">
                        <span class="finding-badge badge-indigo">${escapeHtml(o.category || 'Stat')}</span>
                        <p class="finding-text">${escapeHtml(o.message || o)}</p>
                    </div>
                `).join('');
            }
        }

        // Widget 7: Top Correlations
        if (currentCorrelation) {
            const rankedPairs = currentCorrelation.ranked_pairs || [];
            const posCorr = rankedPairs.find(p => p.pearson_r > 0);
            const negCorr = rankedPairs.slice().reverse().find(p => p.pearson_r < 0);

            const posR = document.getElementById('dash-pos-corr-r');
            const posPair = document.getElementById('dash-pos-corr-pair');
            const posBar = document.getElementById('dash-pos-corr-bar');

            if (posCorr && posR && posPair && posBar) {
                posR.textContent = `+${posCorr.pearson_r.toFixed(3)}`;
                posPair.textContent = `${posCorr.feature_a} & ${posCorr.feature_b}`;
                posBar.style.width = `${Math.abs(posCorr.pearson_r) * 100}%`;
            } else if (posPair) {
                posPair.textContent = 'No positive linear pairs found';
            }

            const negR = document.getElementById('dash-neg-corr-r');
            const negPair = document.getElementById('dash-neg-corr-pair');
            const negBar = document.getElementById('dash-neg-corr-bar');

            if (negCorr && negR && negPair && negBar) {
                negR.textContent = `${negCorr.pearson_r.toFixed(3)}`;
                negPair.textContent = `${negCorr.feature_a} & ${negCorr.feature_b}`;
                negBar.style.width = `${Math.abs(negCorr.pearson_r) * 100}%`;
            } else if (negPair) {
                negPair.textContent = 'No negative linear pairs found';
            }

            const corrBadge = document.getElementById('correlation-count-badge');
            if (corrBadge) corrBadge.textContent = rankedPairs.length;
        }

        // Widget 8: Outlier Summary
        if (currentOutliers) {
            const colMap = currentOutliers.columns || {};
            let totalOutliers = 0;
            const affectedCols = [];

            Object.entries(colMap).forEach(([colName, o]) => {
                const count = o.outlier_count || 0;
                if (count > 0) {
                    totalOutliers += count;
                    affectedCols.push({ colName, count, pct: o.outlier_percentage || 0 });
                }
            });

            const outTotal = document.getElementById('dash-outliers-total');
            const outCols = document.getElementById('dash-outliers-cols-count');
            const outBadge = document.getElementById('dash-outlier-badge-count');
            const outRowPct = document.getElementById('dash-outliers-row-pct');
            if (outTotal) outTotal.textContent = formatNumber(totalOutliers);
            if (outCols) outCols.textContent = `${affectedCols.length}`;
            if (outBadge) outBadge.textContent = totalOutliers;
            
            const rowPct = ov.total_rows ? ((totalOutliers / ov.total_rows) * 100).toFixed(1) : '0.0';
            if (outRowPct) outRowPct.textContent = `${rowPct}%`;

            const outlierColsList = document.getElementById('dash-outliers-columns-list');
            if (outlierColsList) {
                if (affectedCols.length === 0) {
                    outlierColsList.innerHTML = '<p class="text-muted">No extreme statistical outliers detected.</p>';
                } else {
                    outlierColsList.innerHTML = affectedCols.map(a => `
                        <div class="outlier-chip">
                            <span>${escapeHtml(a.colName)}</span>
                            <span class="outlier-chip-badge">${a.count}</span>
                        </div>
                    `).join('');
                }
            }

            const outNavBadge = document.getElementById('outliers-count-badge');
            if (outNavBadge) outNavBadge.textContent = totalOutliers;
        }

        // Widget 9: AI-Generated Executive Summary
        if (currentInsights) {
            const aiSummaryEl = document.getElementById('dash-ai-summary-text');
            const aiKeyFinding = document.getElementById('dash-ai-key-finding');
            const aiKeyRec = document.getElementById('dash-ai-key-recommendation');
            const aiProviderPill = document.getElementById('dash-ai-provider-pill');

            if (aiSummaryEl) aiSummaryEl.innerHTML = currentInsights.executive_summary || 'Analysis compiled successfully.';
            
            const findings = currentInsights.key_findings || [];
            if (aiKeyFinding) aiKeyFinding.textContent = findings[0] || 'Standard variance and metrics observed.';

            const recs = currentInsights.strategic_recommendations || [];
            if (aiKeyRec) aiKeyRec.textContent = recs[0] || 'Continue regular data collection and tracking.';

            if (aiProviderPill) aiProviderPill.textContent = currentInsights.provider_used || 'Gemini 2.5 Flash';
        }

        // Widget 10: Recommended Visualizations (Top 2-3 Plotly Charts)
        renderDashboardTopVisualizations();
    }

    function renderDashboardTopVisualizations() {
        const grid = document.getElementById('dash-recommended-charts-grid');
        if (!grid || allRecommendations.length === 0) return;

        const top2 = allRecommendations.slice(0, 2);
        grid.innerHTML = top2.map((rec, idx) => `
            <div class="dash-chart-card">
                <div class="dash-chart-header">
                    <span class="chart-tag">${escapeHtml(rec.category || 'Distribution')}</span>
                    <h4 class="dash-chart-title">${escapeHtml(rec.title || `Chart #${idx + 1}`)}</h4>
                </div>
                <div class="dash-chart-container" id="dash-top-chart-${idx}"></div>
            </div>
        `).join('');

        top2.forEach((rec, idx) => {
            const plotDiv = document.getElementById(`dash-top-chart-${idx}`);
            if (plotDiv && rec.plotly_spec) {
                const spec = rec.plotly_spec;
                const layout = Object.assign({}, spec.layout || {}, {
                    autosize: true,
                    margin: { l: 40, r: 20, t: 30, b: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#94a3b8', family: 'Plus Jakarta Sans' },
                });
                Plotly.newPlot(plotDiv, spec.data || [], layout, { responsive: true, displayModeBar: false });
            }
        });
    }

    // -------------------------------------------------------------
    // VIEW 2: DATA PREVIEW & EXPLORER
    // -------------------------------------------------------------
    function renderDataPreviewExplorer() {
        renderColumnsProfileTable();
        loadPreviewPage(1);
    }

    function renderColumnsProfileTable() {
        const tbody = document.getElementById('columns-profile-tbody');
        if (!tbody) return;

        let filtered = allColumnsProfile;
        if (activeTypeFilter !== 'all') {
            filtered = filtered.filter(c => (c.classified_type || '').toLowerCase() === activeTypeFilter.toLowerCase());
        }
        if (columnSearchQuery) {
            const q = columnSearchQuery.toLowerCase();
            filtered = filtered.filter(c => {
                const colName = (c.column_name || c.name || '').toLowerCase();
                const dtype = (c.pandas_dtype || '').toLowerCase();
                return colName.includes(q) || dtype.includes(q);
            });
        }

        document.getElementById('column-table-meta').textContent = `Showing ${filtered.length} of ${allColumnsProfile.length} columns`;

        if (filtered.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 20px; color: var(--text-muted);">No matching columns found.</td></tr>';
            return;
        }

        tbody.innerHTML = filtered.map((c, idx) => {
            const colName = c.column_name || c.name;
            const badgeClass = c.classified_type === 'Numerical' ? 'badge-indigo' : c.classified_type === 'Categorical' ? 'badge-cyan' : c.classified_type === 'Datetime' ? 'badge-purple' : 'badge-emerald';
            const samples = (c.sample_values || []).slice(0, 3).map(s => `<code class="code-sample">${escapeHtml(String(s))}</code>`).join(' ');
            
            let statText = '--';
            if (c.classified_type === 'Numerical') {
                const ns = c.numerical_stats || {};
                const meanVal = ns.mean !== undefined ? ns.mean : (c.mean || 0);
                const minVal = ns.min !== undefined ? ns.min : (c.min !== undefined ? c.min : '--');
                const maxVal = ns.max !== undefined ? ns.max : (c.max !== undefined ? c.max : '--');
                statText = `Mean: <strong>${Number(meanVal).toFixed(2)}</strong> &bull; Range: [${minVal}, ${maxVal}]`;
            } else {
                const cs = c.categorical_stats || {};
                const topVal = cs.most_frequent_category || c.mode || 'N/A';
                statText = `Unique: <strong>${c.unique_count || 0}</strong> &bull; Top: ${escapeHtml(String(topVal))}`;
            }

            return `
                <tr>
                    <td>${idx + 1}</td>
                    <td>
                        <strong>${escapeHtml(colName)}</strong>
                        <div style="margin-top: 2px;"><span class="badge ${badgeClass}">${c.classified_type}</span> <span style="font-size:11px; color:var(--text-muted);">${c.pandas_dtype}</span></div>
                    </td>
                    <td>
                        <span>${c.missing_count || 0}</span>
                        <span style="color:${c.missing_percentage > 0 ? '#fbbf24' : 'var(--text-muted)'}; font-size:11px;">(${Number(c.missing_percentage || 0).toFixed(1)}%)</span>
                    </td>
                    <td>
                        <span>${c.unique_count || 0} unique</span>
                    </td>
                    <td style="font-size: 12.5px;">${statText}</td>
                    <td>${samples || '--'}</td>
                    <td style="text-align: center;">
                        <button class="btn btn-outline btn-xs btn-inspect-col" data-col="${escapeHtml(colName)}">Inspect</button>
                    </td>
                </tr>
            `;
        }).join('');

        // Wire inspect buttons
        document.querySelectorAll('.btn-inspect-col').forEach(btn => {
            btn.addEventListener('click', () => {
                const colName = btn.getAttribute('data-col');
                openColumnModal(colName);
            });
        });
    }

    async function loadPreviewPage(page = 1) {
        currentPage = page;
        const thead = document.getElementById('explorer-thead');
        const tbody = document.getElementById('explorer-tbody');
        const pageInd = document.getElementById('page-indicator');

        try {
            const res = await fetch(`/api/preview/${activeDatasetId}?page=${page}&page_size=${pageSize}`);
            const data = await res.json();
            if (!data.success) return;

            const records = data.records || data.rows || [];
            const cols = data.columns || [];
            const totalPages = data.total_pages || (data.pagination && data.pagination.total_pages) || 1;
            const totalRows = data.total_rows || (data.pagination && data.pagination.total_rows) || 0;

            if (pageInd) pageInd.textContent = `Page ${page} of ${totalPages} (${formatNumber(totalRows)} rows)`;

            if (thead) {
                thead.innerHTML = `<tr><th>#</th>` + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + `</tr>`;
            }

            if (tbody) {
                if (records.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="${cols.length + 1}" style="text-align:center; padding: 20px;">No records found.</td></tr>`;
                } else {
                    tbody.innerHTML = records.map((r, i) => {
                        const rowNum = (page - 1) * pageSize + (i + 1);
                        const cells = cols.map(c => `<td>${escapeHtml(String(r[c] !== null && r[c] !== undefined ? r[c] : ''))}</td>`).join('');
                        return `<tr><td><span style="color:var(--text-muted); font-size:11px;">${rowNum}</span></td>${cells}</tr>`;
                    }).join('');
                }
            }
        } catch (e) {
            console.error('Failed to load preview records:', e);
        }
    }

    document.getElementById('btn-prev-page')?.addEventListener('click', () => {
        if (currentPage > 1) loadPreviewPage(currentPage - 1);
    });
    document.getElementById('btn-next-page')?.addEventListener('click', () => {
        loadPreviewPage(currentPage + 1);
    });

    // Column Filters & Search
    document.querySelectorAll('.type-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.type-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeTypeFilter = btn.getAttribute('data-type');
            renderColumnsProfileTable();
        });
    });

    document.getElementById('column-search')?.addEventListener('input', (e) => {
        columnSearchQuery = e.target.value.trim();
        renderColumnsProfileTable();
    });

    // Cleaned Data Toggle in Explorer
    document.getElementById('preview-cleaned-toggle')?.addEventListener('change', (e) => {
        isViewingCleanedData = e.target.checked;
        loadPreviewPage(1);
    });

    // CSV Export button in Explorer
    document.getElementById('preview-download-csv-btn')?.addEventListener('click', (e) => {
        e.preventDefault();
        window.location.href = `/api/cleaning/download/${activeDatasetId}`;
    });

    // -------------------------------------------------------------
    // VIEW 3: DATA QUALITY AUDIT (12-RULE DEFECT ENGINE)
    // -------------------------------------------------------------
    function renderDataQualityAudit() {
        const tbody = document.getElementById('cleaning-issues-tbody');
        if (!tbody) return;

        let filtered = allDetectedIssues;
        if (activeSeverityFilter !== 'all') {
            filtered = filtered.filter(i => (i.severity || '').toLowerCase() === activeSeverityFilter.toLowerCase());
        }

        // Update counts
        const criticalCount = allDetectedIssues.filter(i => i.severity === 'Critical').length;
        const highCount = allDetectedIssues.filter(i => i.severity === 'High').length;
        const medCount = allDetectedIssues.filter(i => i.severity === 'Medium').length;
        const lowCount = allDetectedIssues.filter(i => i.severity === 'Low').length;

        document.getElementById('sev-count-all').textContent = allDetectedIssues.length;
        document.getElementById('sev-count-critical').textContent = criticalCount;
        document.getElementById('sev-count-high').textContent = highCount;
        document.getElementById('sev-count-medium').textContent = medCount;
        document.getElementById('sev-count-low').textContent = lowCount;

        if (filtered.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 25px; color: #34d399;">✓ Zero data quality defects detected under this filter!</td></tr>';
            return;
        }

        tbody.innerHTML = filtered.map((iss, idx) => {
            const sevBadge = iss.severity === 'Critical' ? 'badge-rose' : iss.severity === 'High' ? 'badge-amber' : iss.severity === 'Medium' ? 'badge-indigo' : 'badge-cyan';
            return `
                <tr>
                    <td>${idx + 1}</td>
                    <td><strong>${escapeHtml(iss.column || 'Dataset Wide')}</strong></td>
                    <td>
                        <div style="font-weight: 600;">${escapeHtml(iss.title || iss.issue_type)}</div>
                        <div style="font-size: 11.5px; color: var(--text-secondary);">${escapeHtml(iss.description || '')}</div>
                    </td>
                    <td><code>${formatNumber(iss.affected_rows_count || 0)}</code></td>
                    <td><span class="badge ${sevBadge}">${iss.severity}</span></td>
                    <td style="font-size: 12px; color: #cbd5e1;">${escapeHtml(iss.recommended_action || 'Inspect and clean')}</td>
                </tr>
            `;
        }).join('');
    }

    document.querySelectorAll('.sev-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sev-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeSeverityFilter = btn.getAttribute('data-sev');
            renderDataQualityAudit();
        });
    });

    // -------------------------------------------------------------
    // VIEW 4: CLEANING STUDIO (PIPELINE & EXECUTION)
    // -------------------------------------------------------------
    function renderCleaningStudio(qualData) {
        const previewGrid = document.getElementById('preview-cards-grid');
        const previewPairs = qualData?.preview?.transformation_previews || [];
        const previewCountEl = document.getElementById('preview-pairs-count');

        if (previewCountEl) previewCountEl.textContent = `${previewPairs.length} sample pairs`;

        if (previewGrid) {
            if (previewPairs.length === 0) {
                previewGrid.innerHTML = '<div class="preview-card-placeholder">No value transformation previews needed for this dataset.</div>';
            } else {
                previewGrid.innerHTML = previewPairs.map(p => `
                    <div class="preview-card">
                        <div class="preview-card-col">${escapeHtml(p.column)}</div>
                        <div class="preview-pair-row">
                            <span class="val-orig">${escapeHtml(String(p.original_value))}</span>
                            <span class="val-arrow">&rarr;</span>
                            <span class="val-clean">${escapeHtml(String(p.cleaned_value))}</span>
                        </div>
                        <div class="preview-card-rule">${escapeHtml(p.transformation_type)}</div>
                    </div>
                `).join('');
            }
        }

        // Wire Apply Cleaning
        const applyBtns = [document.getElementById('btn-apply-cleaning'), document.getElementById('btn-apply-cleaning-bottom')];
        applyBtns.forEach(btn => {
            if (!btn) return;
            btn.addEventListener('click', executeCleaningPipeline);
        });

        // Wire Download Cleaned
        const dlBtn = document.getElementById('btn-download-cleaned');
        if (dlBtn) dlBtn.href = `/api/cleaning/download/${activeDatasetId}`;
    }

    async function executeCleaningPipeline() {
        const btn = document.getElementById('btn-apply-cleaning');
        if (btn) btn.disabled = true;

        showToast('Applying configurable cleaning pipeline non-destructively...', '🧹');

        const operations = {
            remove_duplicates: document.getElementById('cfg-remove-duplicates')?.checked ?? true,
            fill_numeric_missing: document.getElementById('cfg-fill-numeric-missing')?.checked ?? true,
            numeric_imputation_strategy: document.getElementById('cfg-numeric-strategy')?.value || 'median',
            fill_categorical_missing: document.getElementById('cfg-fill-categorical-missing')?.checked ?? true,
            categorical_imputation_strategy: document.getElementById('cfg-categorical-strategy')?.value || 'mode',
            standardize_whitespace: document.getElementById('cfg-standardize-whitespace')?.checked ?? true,
            standardize_categorical_casing: document.getElementById('cfg-standardize-categorical')?.checked ?? true,
            convert_numeric_strings: document.getElementById('cfg-convert-numeric-strings')?.checked ?? true,
            convert_datetime_strings: document.getElementById('cfg-convert-datetime-strings')?.checked ?? true,
            handle_invalid_numerical: document.getElementById('cfg-handle-invalid-numerical')?.checked ?? true,
        };

        try {
            const res = await fetch(`/api/cleaning/apply/${activeDatasetId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ operations }),
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Cleaning pipeline failed');

            const sum = data.summary || {};
            currentCleaningSummary = sum;

            // Update Summary Card
            const sumBox = document.getElementById('cleaning-summary-section');
            if (sumBox) sumBox.classList.remove('hidden');

            document.getElementById('clean-kpi-rows-before').textContent = formatNumber(sum.rows_before);
            document.getElementById('clean-kpi-rows-after').textContent = formatNumber(sum.rows_after);
            document.getElementById('clean-kpi-duplicates-removed').textContent = formatNumber(sum.duplicates_removed);
            document.getElementById('clean-kpi-missing-handled').textContent = formatNumber(sum.missing_values_handled);
            document.getElementById('clean-kpi-cols-converted').textContent = formatNumber(sum.columns_converted);
            document.getElementById('clean-kpi-vals-standardized').textContent = formatNumber(sum.values_standardized);

            const chipsWrap = document.getElementById('applied-operations-chips');
            if (chipsWrap && sum.applied_operations) {
                chipsWrap.innerHTML = sum.applied_operations.map(op => `<span class="applied-chip">${escapeHtml(op)}</span>`).join('');
            }

            showToast('Dataset cleaned and saved to data/processed/ successfully!', '✓');

            // Reload full analytics on the cleaned dataset
            setTimeout(() => {
                loadFullDatasetAnalytics(activeDatasetId);
            }, 1000);
        } catch (err) {
            console.error('Cleaning error:', err);
            showToast(`Error: ${err.message}`, '⚠️');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // -------------------------------------------------------------
    // VIEW 5: STATISTICAL ANALYSIS ENGINE
    // -------------------------------------------------------------
    function renderStatisticsView() {
        if (!currentStatistics) return;
        const numStats = currentStatistics.numerical_statistics || {};
        const catStats = currentStatistics.categorical_statistics || {};
        const dtStats = currentStatistics.datetime_statistics || {};

        document.getElementById('stats-kpi-num-count').textContent = Object.keys(numStats).length;
        document.getElementById('stats-kpi-cat-count').textContent = Object.keys(catStats).length;
        document.getElementById('stats-kpi-date-count').textContent = Object.keys(dtStats).length;

        // Render Observations
        const obsGrid = document.getElementById('stats-observations-grid');
        const obs = currentStatistics.statistical_observations || [];
        if (obsGrid) {
            if (obs.length === 0) {
                obsGrid.innerHTML = '<div class="obs-card"><div class="obs-card-icon icon-cyan">ℹ️</div><div class="obs-card-text">Uniform distributions observed across all numerical features.</div></div>';
            } else {
                obsGrid.innerHTML = obs.map(o => `
                    <div class="obs-card">
                        <div class="obs-card-icon icon-indigo">📈</div>
                        <div class="obs-card-text">
                            <strong>${escapeHtml(o.category || 'Observation')}:</strong> ${escapeHtml(o.message || o)}
                        </div>
                    </div>
                `).join('');
            }
        }

        // Render Numerical Table
        const numTbody = document.getElementById('numerical-stats-tbody');
        if (numTbody) {
            const keys = Object.keys(numStats);
            if (keys.length === 0) {
                numTbody.innerHTML = '<tr><td colspan="14" style="text-align:center; padding:20px;">No numerical features present.</td></tr>';
            } else {
                numTbody.innerHTML = keys.map(k => {
                    const s = numStats[k];
                    const ci = s.confidence_interval_95 || [];
                    const ciText = ci.length === 2 ? `[${ci[0].toFixed(1)}, ${ci[1].toFixed(1)}]` : '--';
                    return `
                        <tr>
                            <td><strong>${escapeHtml(k)}</strong></td>
                            <td>${formatNumber(s.count)}</td>
                            <td>${Number(s.mean || 0).toFixed(2)}</td>
                            <td>${Number(s.std || 0).toFixed(2)}</td>
                            <td>${s.min}</td>
                            <td>${Number(s.q1_25 || s.percentile_25 || 0).toFixed(2)}</td>
                            <td><strong>${Number(s.median || s.percentile_50 || 0).toFixed(2)}</strong></td>
                            <td>${Number(s.q3_75 || s.percentile_75 || 0).toFixed(2)}</td>
                            <td>${s.max}</td>
                            <td>${Number(s.iqr || 0).toFixed(2)}</td>
                            <td style="color:${Math.abs(s.skewness || 0) > 1 ? '#fbbf24' : 'inherit'};">${Number(s.skewness || 0).toFixed(2)}</td>
                            <td>${Number(s.kurtosis || 0).toFixed(2)}</td>
                            <td style="font-size:11px; font-family:var(--font-mono);">${ciText}</td>
                            <td style="text-align:center;">
                                <button class="btn btn-outline btn-xs btn-inspect-col" data-col="${escapeHtml(k)}">Inspect</button>
                            </td>
                        </tr>
                    `;
                }).join('');
            }
        }

        // Render Categorical Table
        const catTbody = document.getElementById('categorical-stats-tbody');
        if (catTbody) {
            const keys = Object.keys(catStats);
            if (keys.length === 0) {
                catTbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px;">No categorical features present.</td></tr>';
            } else {
                catTbody.innerHTML = keys.map(k => {
                    const s = catStats[k];
                    const topFreqs = (s.top_categories || []).slice(0, 3).map(t => `<span class="freq-tag">${escapeHtml(t.category)}: ${t.frequency}</span>`).join(' ');
                    return `
                        <tr>
                            <td><strong>${escapeHtml(k)}</strong></td>
                            <td>${formatNumber(s.unique_count)}</td>
                            <td><span class="badge badge-cyan">${escapeHtml(s.mode || 'N/A')}</span></td>
                            <td>${formatNumber(s.mode_frequency)}</td>
                            <td>${Number(s.mode_percentage || 0).toFixed(1)}%</td>
                            <td>${Number(s.entropy_bits || 0).toFixed(2)}</td>
                            <td>${topFreqs || '--'}</td>
                        </tr>
                    `;
                }).join('');
            }
        }
    }

    // -------------------------------------------------------------
    // VIEW 6: CORRELATIONS (HEATMAP & PAIRS)
    // -------------------------------------------------------------
    function renderCorrelationView() {
        if (!currentCorrelation) return;

        // Render Plotly Heatmap
        const heatmapDiv = document.getElementById('plotly-correlation-heatmap');
        if (heatmapDiv && currentHeatmapSpec) {
            const spec = currentHeatmapSpec;
            const layout = Object.assign({}, spec.layout || {}, {
                autosize: true,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#cbd5e1', family: 'Plus Jakarta Sans' },
                margin: { l: 80, r: 40, t: 40, b: 80 },
            });
            Plotly.newPlot(heatmapDiv, spec.data || [], layout, { responsive: true });
        }

        // Render Ranked Pairs Table
        renderCorrelationsTable();

        // Threshold Slider Listener
        const slider = document.getElementById('corr-threshold-slider');
        const display = document.getElementById('corr-threshold-display');
        if (slider) {
            slider.addEventListener('input', (e) => {
                currentCorrThreshold = parseFloat(e.target.value);
                if (display) display.textContent = currentCorrThreshold.toFixed(2);
                renderCorrelationsTable();
            });
        }
    }

    function renderCorrelationsTable() {
        const tbody = document.getElementById('corr-pairs-tbody');
        if (!tbody || !currentCorrelation) return;

        let pairs = currentCorrelation.ranked_pairs || [];
        if (currentCorrThreshold > 0) {
            pairs = pairs.filter(p => Math.abs(p.pearson_r) >= currentCorrThreshold);
        }

        if (pairs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px;">No correlation pairs match threshold.</td></tr>';
            return;
        }

        tbody.innerHTML = pairs.map((p, idx) => {
            const badgeClass = p.strength === 'Strong' ? 'badge-indigo' : p.strength === 'Moderate' ? 'badge-cyan' : 'badge-emerald';
            const dirClass = p.direction === 'Positive' ? 'text-indigo' : 'text-rose';
            return `
                <tr>
                    <td>${idx + 1}</td>
                    <td><strong>${escapeHtml(p.feature_a)}</strong></td>
                    <td><strong>${escapeHtml(p.feature_b)}</strong></td>
                    <td style="font-family:var(--font-mono); font-weight:700;">${p.pearson_r.toFixed(4)}</td>
                    <td><span class="badge ${badgeClass}">${p.strength}</span></td>
                    <td class="${dirClass}"><strong>${p.direction}</strong></td>
                    <td style="font-family:var(--font-mono);">${Math.abs(p.pearson_r).toFixed(4)}</td>
                </tr>
            `;
        }).join('');
    }

    // -------------------------------------------------------------
    // VIEW 7: OUTLIERS (DETECTION & REMEDIATION)
    // -------------------------------------------------------------
    function renderOutliersView(boxplotSpec) {
        if (!currentOutliers) return;

        // Render Multi-Feature Box Plot
        const boxplotDiv = document.getElementById('plotly-outlier-boxplot');
        if (boxplotDiv && boxplotSpec) {
            const layout = Object.assign({}, boxplotSpec.layout || {}, {
                autosize: true,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#cbd5e1', family: 'Plus Jakarta Sans' },
                margin: { l: 50, r: 30, t: 40, b: 60 },
            });
            Plotly.newPlot(boxplotDiv, boxplotSpec.data || [], layout, { responsive: true });
        }

        // Render Outlier Table
        const tbody = document.getElementById('outlier-summary-tbody');
        if (tbody) {
            const colMap = currentOutliers.columns || {};
            const keys = Object.keys(colMap);
            if (keys.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:20px;">No numerical features available for outlier detection.</td></tr>';
            } else {
                tbody.innerHTML = keys.map(k => {
                    const o = colMap[k];
                    const hasOutliers = (o.outlier_count || 0) > 0;
                    return `
                        <tr>
                            <td><strong>${escapeHtml(k)}</strong></td>
                            <td>${formatNumber(o.valid_count)}</td>
                            <td class="${hasOutliers ? 'text-rose' : ''}"><strong>${o.outlier_count || 0}</strong></td>
                            <td>${Number(o.outlier_percentage || 0).toFixed(2)}%</td>
                            <td>${Number(o.lower_threshold || 0).toFixed(2)}</td>
                            <td>${Number(o.upper_threshold || 0).toFixed(2)}</td>
                            <td>${o.confirmed_errors_count || 0}</td>
                            <td>${o.statistical_outliers_count || (o.outlier_count || 0)}</td>
                            <td style="text-align:center;">
                                <button class="btn btn-outline btn-xs btn-drill-outlier" data-col="${escapeHtml(k)}">Drilldown</button>
                            </td>
                        </tr>
                    `;
                }).join('');

                document.querySelectorAll('.btn-drill-outlier').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const colName = btn.getAttribute('data-col');
                        openOutlierModal(colName);
                    });
                });
            }
        }

        // Outlier Method Switcher
        const methodSelect = document.getElementById('outlier-method-select');
        if (methodSelect) {
            methodSelect.addEventListener('change', async (e) => {
                activeOutlierMethod = e.target.value;
                showToast(`Switching outlier detection method to ${activeOutlierMethod}...`, '🎯');
                const outRes = await fetch(`/api/outliers/${activeDatasetId}?method=${activeOutlierMethod}`);
                const outData = await outRes.json();
                if (outData.success) {
                    currentOutliers = outData.outliers;
                    renderOutliersView(outData.boxplot_spec);
                }
            });
        }

        // Outlier Remediation Execution
        document.getElementById('btn-execute-outlier-remediation')?.addEventListener('click', async () => {
            const action = document.getElementById('remed-action-select')?.value || 'remove';
            showToast(`Applying non-destructive outlier remediation (${action})...`, '🛠️');

            try {
                const res = await fetch(`/api/outliers/remediate/${activeDatasetId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action, method: activeOutlierMethod }),
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.error || 'Remediation failed');

                showToast('Outliers remediated successfully! Saved as new processed dataset.', '✓');
                setTimeout(() => {
                    loadFullDatasetAnalytics(activeDatasetId);
                }, 1000);
            } catch (err) {
                console.error('Remediation error:', err);
                showToast(`Error: ${err.message}`, '⚠️');
            }
        });
    }

    // -------------------------------------------------------------
    // VIEW 8: VISUALIZATIONS (RECOMMENDATIONS & BUILDER)
    // -------------------------------------------------------------
    function renderVisualizationsView() {
        renderChartRecommendations();
        populateCustomChartBuilderOptions();
    }

    function renderChartRecommendations() {
        const grid = document.getElementById('viz-recommendations-grid');
        if (!grid) return;

        let filtered = allRecommendations;
        if (activeVizFilter !== 'all') {
            filtered = filtered.filter(r => (r.category || '').toLowerCase() === activeVizFilter.toLowerCase());
        }

        document.getElementById('viz-count-all').textContent = allRecommendations.length;
        document.getElementById('viz-count-dist').textContent = allRecommendations.filter(r => (r.category || '').toLowerCase() === 'distribution').length;
        document.getElementById('viz-count-rel').textContent = allRecommendations.filter(r => (r.category || '').toLowerCase() === 'relationship').length;
        document.getElementById('viz-count-cat').textContent = allRecommendations.filter(r => (r.category || '').toLowerCase() === 'categorical').length;
        document.getElementById('viz-count-temp').textContent = allRecommendations.filter(r => (r.category || '').toLowerCase() === 'temporal').length;

        if (filtered.length === 0) {
            grid.innerHTML = '<div class="preview-card-placeholder">No chart recommendations match the selected category filter.</div>';
            return;
        }

        grid.innerHTML = filtered.map((rec, idx) => `
            <div class="dash-chart-card">
                <div class="dash-chart-header">
                    <div>
                        <span class="chart-tag">${escapeHtml(rec.category || 'Chart')}</span>
                        <h4 class="dash-chart-title" style="margin-top: 4px;">${escapeHtml(rec.title || `Chart #${idx + 1}`)}</h4>
                    </div>
                </div>
                <div class="dash-chart-container" id="viz-rec-plot-${idx}"></div>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">${escapeHtml(rec.rationale || '')}</p>
            </div>
        `).join('');

        filtered.forEach((rec, idx) => {
            const plotDiv = document.getElementById(`viz-rec-plot-${idx}`);
            if (plotDiv && rec.plotly_spec) {
                const spec = rec.plotly_spec;
                const layout = Object.assign({}, spec.layout || {}, {
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#cbd5e1', family: 'Plus Jakarta Sans' },
                    margin: { l: 45, r: 25, t: 35, b: 45 },
                });
                Plotly.newPlot(plotDiv, spec.data || [], layout, { responsive: true });
            }
        });
    }

    document.querySelectorAll('.viz-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.viz-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeVizFilter = btn.getAttribute('data-category');
            renderChartRecommendations();
        });
    });

    function populateCustomChartBuilderOptions() {
        const xSelect = document.getElementById('custom-x-col');
        const ySelect = document.getElementById('custom-y-col');
        const colorSelect = document.getElementById('custom-color-col');

        if (!xSelect || allColumnsProfile.length === 0) return;

        const optionsHtml = allColumnsProfile.map(c => {
            const colName = c.column_name || c.name;
            return `<option value="${escapeHtml(colName)}">${escapeHtml(colName)} (${c.classified_type})</option>`;
        }).join('');

        xSelect.innerHTML = `<option value="">Select feature...</option>` + optionsHtml;
        if (ySelect) ySelect.innerHTML = `<option value="">None (Count / Frequency)</option>` + optionsHtml;
        if (colorSelect) colorSelect.innerHTML = `<option value="">None</option>` + optionsHtml;

        document.getElementById('btn-generate-custom-chart')?.addEventListener('click', generateCustomChart);
    }

    async function generateCustomChart() {
        const chartType = document.getElementById('custom-chart-type')?.value || 'bar';
        const xCol = document.getElementById('custom-x-col')?.value;
        const yCol = document.getElementById('custom-y-col')?.value || null;
        const colorCol = document.getElementById('custom-color-col')?.value || null;
        const aggregation = document.getElementById('custom-aggregation')?.value || 'none';

        if (!xCol) {
            showToast('Please select an X-Axis feature.', '⚠️');
            return;
        }

        showToast('Generating custom visualization...', '🎨');

        try {
            const res = await fetch(`/api/visualization/custom/${activeDatasetId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chart_type: chartType,
                    x_col: xCol,
                    y_col: yCol,
                    color_col: colorCol,
                    aggregation: aggregation,
                }),
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to generate chart');

            const resultContainer = document.getElementById('custom-chart-result-container');
            const plotDiv = document.getElementById('plotly-custom-chart');
            const titleDisplay = document.getElementById('custom-chart-title-display');

            if (resultContainer) resultContainer.classList.remove('hidden');
            if (titleDisplay) titleDisplay.textContent = `${chartType.toUpperCase()} of ${xCol}${yCol ? ' vs ' + yCol : ''}`;

            if (plotDiv && data.spec) {
                const spec = data.spec;
                const layout = Object.assign({}, spec.layout || {}, {
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#cbd5e1', family: 'Plus Jakarta Sans' },
                });
                Plotly.newPlot(plotDiv, spec.data || [], layout, { responsive: true });
            }

            showToast('Custom chart generated successfully!', '✓');
        } catch (err) {
            console.error('Custom chart error:', err);
            showToast(`Error: ${err.message}`, '⚠️');
        }
    }

    // -------------------------------------------------------------
    // VIEW 9: AI INSIGHTS ENGINE
    // -------------------------------------------------------------
    function renderAiInsightsView() {
        if (!currentInsights) return;

        const setContent = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = text || 'Analysis metric verified.';
        };

        setContent('content-executive-summary', currentInsights.executive_summary);
        setContent('content-key-findings', (currentInsights.key_findings || []).map(f => `&bull; ${escapeHtml(f)}`).join('<br><br>'));
        setContent('content-correlations-patterns', (currentInsights.important_relationships || []).map(f => `&bull; ${escapeHtml(f)}`).join('<br><br>'));
        setContent('content-data-quality-concerns', (currentInsights.data_quality_concerns || []).map(f => `&bull; ${escapeHtml(f)}`).join('<br><br>'));
        setContent('content-potential-outliers', (currentInsights.anomalies_and_risks || currentInsights.potential_outliers || []).map(f => `&bull; ${escapeHtml(f)}`).join('<br><br>'));
        setContent('content-business-recommendations', (currentInsights.strategic_recommendations || currentInsights.business_recommendations || []).map(f => `&bull; ${escapeHtml(f)}`).join('<br><br>'));

        // Regenerate Button
        document.getElementById('btn-regenerate-insights')?.addEventListener('click', async () => {
            const provider = document.getElementById('ai-provider-select')?.value;
            showToast(`Regenerating AI insights using ${provider}...`, '✨');
            try {
                const res = await fetch(`/api/insights/generate/${activeDatasetId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ provider }),
                });
                const data = await res.json();
                if (data.success) {
                    currentInsights = data.insights;
                    renderAiInsightsView();
                    renderDashboardExecutiveWidgets();
                    showToast('AI insights successfully refreshed!', '✓');
                }
            } catch (err) {
                showToast(`AI generation error: ${err.message}`, '⚠️');
            }
        });

        // Inspect Raw Context Modal
        const btnInspect = document.getElementById('btn-inspect-raw-context');
        const dashBtnInspect = document.getElementById('dash-btn-inspect-context');
        [btnInspect, dashBtnInspect].forEach(b => {
            if (!b) return;
            b.addEventListener('click', openRawContextModal);
        });
    }

    async function openRawContextModal() {
        const modal = document.getElementById('raw-context-modal');
        const codeEl = document.getElementById('raw-context-json-content');
        if (!modal) return;

        modal.classList.remove('hidden');
        if (codeEl) codeEl.textContent = 'Fetching exact structured Python context...';

        try {
            const res = await fetch(`/api/insights/context/${activeDatasetId}`);
            const data = await res.json();
            if (data.success && codeEl) {
                codeEl.textContent = JSON.stringify(data.analysis_context, null, 2);
            }
        } catch (e) {
            if (codeEl) codeEl.textContent = 'Failed to fetch analysis context.';
        }
    }

    document.getElementById('modal-raw-context-close-btn')?.addEventListener('click', () => {
        document.getElementById('raw-context-modal')?.classList.add('hidden');
    });

    // -------------------------------------------------------------
    // VIEW 10: ASK DATASET (CONVERSATIONAL Q&A ASSISTANT)
    // -------------------------------------------------------------
    async function loadChatSuggestions() {
        const chipsContainer = document.getElementById('chat-suggested-chips');
        if (!chipsContainer) return;

        try {
            const res = await fetch(`/api/chat/suggested-questions/${activeDatasetId}`);
            const data = await res.json();
            if (data.success && data.suggestions) {
                chipsContainer.innerHTML = data.suggestions.map(s => `
                    <button class="chip-btn" data-query="${escapeHtml(s)}">${escapeHtml(s)}</button>
                `).join('');

                chipsContainer.querySelectorAll('.chip-btn').forEach(b => {
                    b.addEventListener('click', () => {
                        const q = b.getAttribute('data-query');
                        const input = document.getElementById('chat-query-input');
                        if (input) {
                            input.value = q;
                            sendChatQuery();
                        }
                    });
                });
            }
        } catch (e) {
            console.error('Failed to load chat suggestions:', e);
        }
    }

    async function sendChatQuery() {
        const input = document.getElementById('chat-query-input');
        const sendBtn = document.getElementById('chat-send-btn');
        const loader = document.getElementById('chat-loader');
        const container = document.getElementById('chat-messages-container');

        if (!input || !container) return;
        const query = input.value.trim();
        if (!query || isChatStreaming) return;

        // Append User Message
        const userMsgHtml = `
            <div class="chat-message message-user">
                <div class="message-avatar">👤</div>
                <div class="message-content-wrap">
                    <div class="message-sender-name">You</div>
                    <div class="message-text">${escapeHtml(query)}</div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', userMsgHtml);
        input.value = '';
        container.scrollTop = container.scrollHeight;

        // Append Loading Agent Message
        const tempMsgId = `agent-msg-${Date.now()}`;
        const agentLoadingHtml = `
            <div class="chat-message message-agent" id="${tempMsgId}">
                <div class="message-avatar">⚡</div>
                <div class="message-content-wrap">
                    <div class="message-sender-name">AI Data Analyst</div>
                    <div class="message-text">
                        <span class="spinner-sm"></span> Formulating analysis plan & executing deterministic tool...
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', agentLoadingHtml);
        container.scrollTop = container.scrollHeight;

        isChatStreaming = true;
        if (sendBtn) sendBtn.disabled = true;

        try {
            const res = await fetch('/api/chat/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: activeDatasetId,
                    query: query,
                }),
            });
            const data = await res.json();
            const msgEl = document.getElementById(tempMsgId);

            if (msgEl) {
                const answerText = data.success ? (data.answer || 'Analysis complete.') : (`⚠️ ${data.error || 'Failed to process question'}`);
                const plan = data.plan || {};
                const plotId = `plot-${Date.now()}`;

                let planHtml = '';
                if (plan.steps && plan.steps.length > 0) {
                    planHtml = `
                        <div class="analysis-plan-box">
                            <div class="analysis-plan-header">
                                <span class="badge-plan-tool">Tool: ${escapeHtml(plan.tool || 'Python Engine')}</span>
                                <span class="plan-steps-count">${plan.steps.length} Step Plan</span>
                            </div>
                            <ul class="plan-steps-list">
                                ${plan.steps.map(s => `<li class="plan-step-item"><span class="step-check">✓</span> <span class="step-text">${escapeHtml(s)}</span></li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                let plotHtml = '';
                if (data.plotly_spec) {
                    plotHtml = `<div class="chat-plot-box" id="${plotId}"></div>`;
                }

                msgEl.querySelector('.message-text').innerHTML = `
                    ${planHtml}
                    <div class="answer-body-markdown">${answerText.replace(/\n/g, '<br>')}</div>
                    ${plotHtml}
                `;

                if (data.plotly_spec) {
                    setTimeout(() => {
                        const pDiv = document.getElementById(plotId);
                        if (pDiv) {
                            const spec = data.plotly_spec;
                            const layout = Object.assign({}, spec.layout || {}, {
                                autosize: true,
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: '#cbd5e1', family: 'Plus Jakarta Sans' },
                            });
                            Plotly.newPlot(pDiv, spec.data || [], layout, { responsive: true });
                        }
                    }, 50);
                }
            }
        } catch (err) {
            console.error('Chat error:', err);
            const msgEl = document.getElementById(tempMsgId);
            if (msgEl) {
                msgEl.querySelector('.message-text').textContent = `⚠️ Error: ${err.message}`;
            }
        } finally {
            isChatStreaming = false;
            if (sendBtn) sendBtn.disabled = false;
            container.scrollTop = container.scrollHeight;
        }
    }

    document.getElementById('chat-send-btn')?.addEventListener('click', sendChatQuery);
    document.getElementById('chat-query-input')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendChatQuery();
    });

    document.getElementById('btn-clear-chat-history')?.addEventListener('click', async () => {
        try {
            await fetch(`/api/chat/clear/${activeDatasetId}`, { method: 'POST' });
            const container = document.getElementById('chat-messages-container');
            if (container) {
                container.innerHTML = `
                    <div class="chat-message message-agent">
                        <div class="message-avatar">⚡</div>
                        <div class="message-content-wrap">
                            <div class="message-sender-name">AI Data Analyst</div>
                            <div class="message-text">Chat history cleared. Ready for your next question!</div>
                        </div>
                    </div>
                `;
            }
            showToast('Chat history cleared.', '✓');
        } catch (e) {}
    });

    // -------------------------------------------------------------
    // VIEW 11: REPORTS STUDIO (EXECUTIVE REPORT COMPILER)
    // -------------------------------------------------------------
    async function loadExecutiveReport(isManualTrigger = false) {
        const htmlPreview = document.getElementById('report-html-preview');
        const mdContent = document.getElementById('report-md-content');
        if (!htmlPreview) return;

        htmlPreview.innerHTML = `
            <div class="report-loading-state">
                <div class="spinner"></div>
                <p>Compiling 9-section automated analysis report for <strong>${escapeHtml(activeDatasetId)}</strong>...</p>
            </div>
        `;

        try {
            const res = await fetch(`/api/report/generate/${activeDatasetId}`);
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to generate report');

            currentReport = data;

            // Render HTML preview
            if (data.html) {
                htmlPreview.innerHTML = data.html;
                // Execute inline chart scripts within injected HTML
                const scripts = htmlPreview.querySelectorAll('script');
                scripts.forEach(s => {
                    try {
                        const newScript = document.createElement('script');
                        newScript.textContent = s.textContent;
                        document.body.appendChild(newScript);
                        setTimeout(() => newScript.remove(), 100);
                    } catch (e) {
                        console.warn('Inline chart script execution error:', e);
                    }
                });
            }

            // Render Markdown preview
            if (mdContent && data.markdown) {
                mdContent.textContent = data.markdown;
            }

            // Wire Download Links
            const dlMd = document.getElementById('btn-download-report-md');
            const dlHtml = document.getElementById('btn-download-report-html');
            const dlJson = document.getElementById('btn-download-report-json');
            if (dlMd) dlMd.href = `/api/report/download/${activeDatasetId}?format=md`;
            if (dlHtml) dlHtml.href = `/api/report/download/${activeDatasetId}?format=html`;
            if (dlJson) dlJson.href = `/api/report/download/${activeDatasetId}?format=json`;

            if (isManualTrigger) {
                showToast('Executive Analysis Report generated successfully!', '✓');
            }
        } catch (err) {
            console.error('Report error:', err);
            htmlPreview.innerHTML = `<div class="error-card"><p>Failed to generate report: ${err.message}</p></div>`;
        }
    }

    // Explicit "Generate Report" Button Click
    document.getElementById('btn-generate-report')?.addEventListener('click', () => {
        loadExecutiveReport(true);
    });

    // Report Mode Switcher (HTML Presentation vs Markdown Source)
    document.getElementById('btn-report-view-html')?.addEventListener('click', () => {
        document.getElementById('btn-report-view-html').classList.add('active');
        document.getElementById('btn-report-view-md').classList.remove('active');
        document.getElementById('report-html-preview').classList.remove('hidden');
        document.getElementById('report-md-preview').classList.add('hidden');
    });

    document.getElementById('btn-report-view-md')?.addEventListener('click', () => {
        document.getElementById('btn-report-view-md').classList.add('active');
        document.getElementById('btn-report-view-html').classList.remove('active');
        document.getElementById('report-md-preview').classList.remove('hidden');
        document.getElementById('report-html-preview').classList.add('hidden');
    });

    // Copy Markdown to Clipboard
    document.getElementById('btn-copy-report-md')?.addEventListener('click', () => {
        if (currentReport && currentReport.markdown) {
            navigator.clipboard.writeText(currentReport.markdown);
            showToast('Executive Report Markdown copied to clipboard!', '📋');
        }
    });

    // Print to PDF
    document.getElementById('btn-print-report')?.addEventListener('click', () => {
        window.print();
    });

    // -------------------------------------------------------------
    // Modal Handlers (Column Drilldown & Outlier Inspector)
    // -------------------------------------------------------------
    function openColumnModal(colName) {
        const modal = document.getElementById('column-modal-backdrop');
        const titleEl = document.getElementById('modal-col-name');
        const typeEl = document.getElementById('modal-col-type');
        const dtypeEl = document.getElementById('modal-col-dtype');
        const bodyEl = document.getElementById('modal-col-body');
        if (!modal) return;

        const col = allColumnsProfile.find(c => (c.column_name === colName || c.name === colName));
        if (!col) return;

        const actualColName = col.column_name || col.name;
        titleEl.textContent = actualColName;
        typeEl.textContent = col.classified_type;
        dtypeEl.textContent = col.pandas_dtype;

        const samplesHtml = (col.sample_values || []).map(s => `<span class="val-pill">${escapeHtml(String(s))}</span>`).join(' ');

        bodyEl.innerHTML = `
            <div class="modal-metric-grid">
                <div class="kpi-card"><div class="kpi-label">Valid Values</div><div class="kpi-value">${formatNumber(col.non_null_count)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Missing Count</div><div class="kpi-value text-amber">${formatNumber(col.missing_count)} (${Number(col.missing_percentage || 0).toFixed(1)}%)</div></div>
                <div class="kpi-card"><div class="kpi-label">Unique Values</div><div class="kpi-value">${formatNumber(col.unique_count)}</div></div>
            </div>
            <div style="margin-top: 16px;">
                <h4 style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">Sample Records:</h4>
                <div style="display:flex; flex-wrap:wrap; gap:6px;">${samplesHtml || 'None'}</div>
            </div>
        `;

        modal.classList.remove('hidden');
    }

    document.getElementById('modal-close-btn')?.addEventListener('click', () => {
        document.getElementById('column-modal-backdrop')?.classList.add('hidden');
    });

    async function openOutlierModal(colName) {
        const modal = document.getElementById('outlier-inspector-modal');
        const titleEl = document.getElementById('modal-outlier-title');
        const tbody = document.getElementById('modal-outlier-tbody');
        if (!modal || !tbody) return;

        titleEl.textContent = `Outlier Instances: ${colName}`;
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px;">Fetching row-level outlier audit records...</td></tr>';
        modal.classList.remove('hidden');

        try {
            const res = await fetch(`/api/outliers/${activeDatasetId}/column/${encodeURIComponent(colName)}?method=${activeOutlierMethod}`);
            const data = await res.json();
            const resData = data.result || {};
            const outliers = resData.outlier_instances || [];

            if (outliers.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px; color:#34d399;">✓ No outlier rows detected in this feature under selected threshold.</td></tr>';
            } else {
                tbody.innerHTML = outliers.slice(0, 50).map(o => `
                    <tr>
                        <td><code>#${o.row_index}</code></td>
                        <td><strong>${o.value}</strong></td>
                        <td><span class="badge ${o.bound_violated === 'Upper Bound' ? 'badge-rose' : 'badge-amber'}">${o.bound_violated}</span></td>
                        <td>${Number(o.threshold || 0).toFixed(2)}</td>
                        <td>${Number(o.deviation || 0).toFixed(2)}</td>
                        <td><span class="badge ${o.classification === 'Confirmed Data Error' ? 'badge-rose' : 'badge-indigo'}">${o.classification || 'Outlier'}</span></td>
                        <td style="font-size: 11.5px; color: var(--text-secondary);">${escapeHtml(o.rationale || '')}</td>
                    </tr>
                `).join('');
            }
        } catch (e) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px; color:#fb7185;">Failed to load outlier instances.</td></tr>';
        }
    }

    document.getElementById('modal-outlier-close-btn')?.addEventListener('click', () => {
        document.getElementById('outlier-inspector-modal')?.classList.add('hidden');
    });

    // -------------------------------------------------------------
    // Sample Datasets Loader
    // -------------------------------------------------------------
    async function loadSampleDataset(sampleType) {
        showToast(`Loading bundled sample ${sampleType} dataset...`, '⚡');
        try {
            const res = await fetch(`/api/sample/${sampleType}`, { method: 'POST' });
            const data = await res.json();
            if (data.success && data.dataset_id) {
                showToast(`Sample dataset '${data.original_filename}' loaded!`, '✓');
                await loadDatasetList();
                loadFullDatasetAnalytics(data.dataset_id);
            }
        } catch (err) {
            showToast(`Failed to load sample: ${err.message}`, '⚠️');
        }
    }

    btnSampleEcommerce?.addEventListener('click', () => loadSampleDataset('ecommerce'));
    btnSampleEmployee?.addEventListener('click', () => loadSampleDataset('employee'));
    btnLoadSampleErr?.addEventListener('click', () => loadSampleDataset('ecommerce'));

    // Dataset Selector Switcher
    datasetSelector?.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val && val !== 'undefined') loadFullDatasetAnalytics(val);
    });

    // -------------------------------------------------------------
    // App Initialization
    // -------------------------------------------------------------
    async function init() {
        await loadDatasetList();

        let targetId = activeDatasetId;

        // 1. Check URL parameters
        if (!targetId || targetId === 'undefined') {
            try {
                const urlParams = new URLSearchParams(window.location.search);
                const queryDsId = urlParams.get('dataset_id');
                if (queryDsId && queryDsId !== 'undefined') {
                    targetId = queryDsId;
                }
            } catch (e) {}
        }

        // 2. Check localStorage
        if (!targetId || targetId === 'undefined') {
            try {
                const storedId = localStorage.getItem('active_dataset_id');
                if (storedId && storedId !== 'undefined') {
                    // Check if storedId is present in the selector options
                    let exists = false;
                    if (datasetSelector && datasetSelector.options) {
                        for (let i = 0; i < datasetSelector.options.length; i++) {
                            if (datasetSelector.options[i].value === storedId) {
                                exists = true;
                                break;
                            }
                        }
                    }
                    if (exists || !datasetSelector || datasetSelector.options.length <= 1) {
                        targetId = storedId;
                    }
                }
            } catch (e) {}
        }

        // 3. Auto-load first uploaded dataset from selector if none selected yet
        if (!targetId || targetId === 'undefined') {
            const firstOpt = datasetSelector?.options[1]?.value;
            if (firstOpt && firstOpt !== 'undefined') {
                targetId = firstOpt;
            }
        }

        // 4. Load dataset or fallback to bundled sample
        if (targetId && targetId !== 'undefined') {
            loadFullDatasetAnalytics(targetId);
        } else {
            loadSampleDataset('ecommerce');
        }
    }

    init();
});

/**
 * Main Client-Side Controller for AI Data Analyst Agent.
 * Handles drag-and-drop file upload, comprehensive validation feedback,
 * and instant rendering of the Ingestion Summary & First 10 Rows Preview.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const filePreviewBar = document.getElementById('file-preview-bar');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadLoader = document.getElementById('upload-loader');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const alertBox = document.getElementById('alert-box');
    const alertMessage = document.getElementById('alert-message');
    const sampleEcommerceBtn = document.getElementById('load-sample-ecommerce');
    const sampleEmployeeBtn = document.getElementById('load-sample-employee');

    // Ingestion Summary Elements
    const ingestionSummarySection = document.getElementById('ingestion-summary-section');
    const summaryFileName = document.getElementById('summary-file-name');
    const summaryDatasetId = document.getElementById('summary-dataset-id');
    const summaryEncoding = document.getElementById('summary-encoding');
    const summaryDelimiter = document.getElementById('summary-delimiter');
    const sumMetricRows = document.getElementById('sum-metric-rows');
    const sumMetricCols = document.getElementById('sum-metric-cols');
    const sumMetricMemory = document.getElementById('sum-metric-memory');
    const sumMetricMissing = document.getElementById('sum-metric-missing');
    const sumMetricMissingPct = document.getElementById('sum-metric-missing-pct');
    const summaryColumnsTbody = document.getElementById('summary-columns-tbody');
    const summaryPreviewThead = document.getElementById('summary-preview-thead');
    const summaryPreviewTbody = document.getElementById('summary-preview-tbody');
    const goToDashboardBtn = document.getElementById('go-to-dashboard-btn');

    let selectedFile = null;

    // Helper: Format bytes
    function formatBytes(bytes, decimals = 2) {
        if (!bytes || bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
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

    // Show Alert
    function showAlert(message, type = 'error') {
        if (!alertBox || !alertMessage) return;
        alertBox.className = `alert-box alert-${type}`;
        alertMessage.textContent = message;
        alertBox.classList.remove('hidden');
    }

    function hideAlert() {
        if (alertBox) alertBox.classList.add('hidden');
    }

    // File Selection Handler
    function handleFile(file) {
        hideAlert();
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.csv')) {
            showAlert(`Invalid file format '${file.name}'. Only CSV files (.csv extension) are supported.`, 'error');
            return;
        }

        if (file.size === 0) {
            showAlert('Selected file is empty (0 bytes). Please choose a valid CSV file with data.', 'error');
            return;
        }

        // 50MB limit check
        if (file.size > 50 * 1024 * 1024) {
            showAlert(`File size (${(file.size / (1024*1024)).toFixed(1)} MB) exceeds the 50MB maximum limit.`, 'error');
            return;
        }

        selectedFile = file;
        if (selectedFileName) selectedFileName.textContent = file.name;
        if (selectedFileSize) selectedFileSize.textContent = formatBytes(file.size);
        if (filePreviewBar) filePreviewBar.classList.remove('hidden');
    }

    // Dropzone Event Listeners
    if (dropzone && fileInput) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('drag-over');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
    }

    // Render Ingestion Summary
    function renderIngestionSummary(data) {
        const summary = data.summary;
        const profile = data.profile || {};
        const overview = profile.overview || {};

        if (!summary || !ingestionSummarySection) return;

        if (summaryFileName) summaryFileName.textContent = summary.file_name || data.original_filename;
        if (summaryDatasetId) summaryDatasetId.textContent = data.dataset_id;
        if (summaryEncoding) summaryEncoding.textContent = summary.encoding || 'utf-8';
        if (summaryDelimiter) summaryDelimiter.textContent = summary.delimiter === '\t' ? '\\t (tab)' : summary.delimiter;

        if (sumMetricRows) sumMetricRows.textContent = (summary.total_rows || 0).toLocaleString();
        if (sumMetricCols) sumMetricCols.textContent = (summary.total_columns || 0).toLocaleString();
        if (sumMetricMemory) sumMetricMemory.textContent = summary.memory_usage_formatted || '0 KB';

        const totalMissing = overview.total_missing_cells !== undefined ? overview.total_missing_cells : (
            summary.columns.reduce((acc, col) => acc + (col.null_count || 0), 0)
        );
        const missingPct = overview.missing_cells_percentage !== undefined ? overview.missing_cells_percentage : (
            summary.total_rows > 0 && summary.total_columns > 0 ? (totalMissing / (summary.total_rows * summary.total_columns) * 100).toFixed(2) : 0
        );

        if (sumMetricMissing) sumMetricMissing.textContent = totalMissing.toLocaleString();
        if (sumMetricMissingPct) sumMetricMissingPct.textContent = `${missingPct}% missing data rate`;

        // Render Ingested Columns Table
        if (summaryColumnsTbody && summary.columns) {
            summaryColumnsTbody.innerHTML = summary.columns.map((col, idx) => {
                return `
                    <tr>
                        <td style="color: var(--text-muted); font-family: var(--font-mono);">${idx + 1}</td>
                        <td style="font-weight: 600; color: var(--text-primary); font-family: var(--font-mono);">${escapeHtml(col.name)}</td>
                        <td><code>${escapeHtml(col.pandas_dtype)}</code></td>
                        <td><span class="type-pill type-badge type-${col.inferred_type}">${escapeHtml(col.inferred_type)}</span></td>
                        <td><strong>${(col.non_null_count || 0).toLocaleString()}</strong></td>
                        <td>${(col.null_count || 0).toLocaleString()}</td>
                        <td>${col.null_percentage}%</td>
                    </tr>
                `;
            }).join('');
        }

        // Render First 10 Rows Preview Table
        if (summaryPreviewThead && summaryPreviewTbody && summary.preview_first_10_rows) {
            const cols = summary.column_names || [];
            
            // Header
            summaryPreviewThead.innerHTML = `
                <tr>
                    <th style="width: 40px;">#</th>
                    ${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}
                </tr>
            `;

            // Body
            summaryPreviewTbody.innerHTML = summary.preview_first_10_rows.map((row) => {
                const cells = cols.map((col) => {
                    const val = row[col];
                    if (val === null || val === undefined) {
                        return `<td><span class="null-badge">NULL</span></td>`;
                    }
                    return `<td>${escapeHtml(val)}</td>`;
                }).join('');

                return `<tr><td style="color: var(--text-muted); font-family: var(--font-mono);">${row['#']}</td>${cells}</tr>`;
            }).join('');
        }

        // Set Dashboard & Chat Links and persist in localStorage
        if (data.dataset_id) {
            try {
                localStorage.setItem('active_dataset_id', data.dataset_id);
            } catch (e) {}

            const navDash = document.getElementById('nav-dashboard');
            if (navDash) navDash.href = `/dashboard?dataset_id=${encodeURIComponent(data.dataset_id)}`;
            const navChat = document.getElementById('nav-chat');
            if (navChat) navChat.href = `/chat?dataset_id=${encodeURIComponent(data.dataset_id)}`;
        }

        if (goToDashboardBtn && data.dataset_id) {
            goToDashboardBtn.href = `/dashboard?dataset_id=${encodeURIComponent(data.dataset_id)}`;
        }

        // Show section and smooth scroll
        ingestionSummarySection.classList.remove('hidden');
        ingestionSummarySection.scrollIntoView({ behavior: 'smooth' });
    }

    // Upload Button Click Action
    if (uploadBtn) {
        uploadBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                showAlert('Please select a CSV file first.', 'error');
                return;
            }

            hideAlert();
            uploadBtn.disabled = true;
            if (uploadLoader) uploadLoader.classList.remove('hidden');
            if (progressContainer) progressContainer.classList.remove('hidden');
            if (progressBar) progressBar.style.width = '60%';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (progressBar) progressBar.style.width = '100%';

                if (response.ok && data.success) {
                    showAlert(`Dataset '${selectedFile.name}' ingested successfully! Review summary below.`, 'success');
                    renderIngestionSummary(data);
                    uploadBtn.disabled = false;
                    if (uploadLoader) uploadLoader.classList.add('hidden');
                } else {
                    showAlert(data.error || 'Upload failed. Please check your CSV file.', 'error');
                    uploadBtn.disabled = false;
                    if (uploadLoader) uploadLoader.classList.add('hidden');
                    if (progressContainer) progressContainer.classList.add('hidden');
                }
            } catch (err) {
                showAlert(`Network error during upload: ${err.message}`, 'error');
                uploadBtn.disabled = false;
                if (uploadLoader) uploadLoader.classList.add('hidden');
                if (progressContainer) progressContainer.classList.add('hidden');
            }
        });
    }

    // Sample Dataset Quick Loader Helper
    async function loadSample(sampleType, btnElement) {
        hideAlert();
        const origText = btnElement.innerHTML;
        btnElement.disabled = true;
        btnElement.innerHTML = '<span>⏳ Ingesting sample...</span>';

        try {
            const response = await fetch(`/api/sample/${sampleType}`, {
                method: 'POST',
            });
            const data = await response.json();

            if (response.ok && data.success) {
                showAlert(`Sample dataset '${data.original_filename}' ingested successfully!`, 'success');
                renderIngestionSummary(data);
                btnElement.disabled = false;
                btnElement.innerHTML = origText;
            } else {
                showAlert(data.error || 'Failed to load sample dataset.', 'error');
                btnElement.disabled = false;
                btnElement.innerHTML = origText;
            }
        } catch (err) {
            showAlert(`Network error: ${err.message}`, 'error');
            btnElement.disabled = false;
            btnElement.innerHTML = origText;
        }
    }

    if (sampleEcommerceBtn) {
        sampleEcommerceBtn.addEventListener('click', () => loadSample('ecommerce', sampleEcommerceBtn));
    }
    if (sampleEmployeeBtn) {
        sampleEmployeeBtn.addEventListener('click', () => loadSample('employee', sampleEmployeeBtn));
    }

    // Initialize navbar links if an active dataset exists in storage
    try {
        const savedDatasetId = localStorage.getItem('active_dataset_id');
        if (savedDatasetId) {
            const navDash = document.getElementById('nav-dashboard');
            if (navDash) navDash.href = `/dashboard?dataset_id=${encodeURIComponent(savedDatasetId)}`;
            const navChat = document.getElementById('nav-chat');
            if (navChat) navChat.href = `/chat?dataset_id=${encodeURIComponent(savedDatasetId)}`;
        }
    } catch (e) {}
});

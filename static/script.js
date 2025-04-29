let calibratedParams = null;

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('excel-file');
    const modelSelect = document.getElementById('model');
    const loader = document.getElementById('calibration-loader');
    const formData = new FormData();
    
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect.value);
    
    try {
        loader.style.display = 'block'; // Show loader
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (response.ok) {
            if (data.multiple_sheets) {
                showSheetSelector(data.sheets, fileInput.files[0], modelSelect.value);
            } else {
                calibratedParams = data;
                displayCalibratedParams(data);
                document.querySelector('.parameters-section').style.display = 'block';
                document.querySelector('.pricing-section').style.display = 'block';
            }
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error uploading file: ' + error);
    } finally {
        loader.style.display = 'none'; // Hide loader
    }
});

document.getElementById('pricing-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!calibratedParams) {
        alert('Please upload and calibrate model first');
        return;
    }
    
    const pricingParams = {
        ...calibratedParams,
        S0: parseFloat(document.getElementById('spot').value),
        K: parseFloat(document.getElementById('strike').value),
        T: parseFloat(document.getElementById('maturity').value)
    };
    
    try {
        const response = await fetch('/price', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(pricingParams)
        });
        
        const data = await response.json();
        if (response.ok) {
            document.getElementById('call-price').textContent = data.call.toFixed(4);
            document.getElementById('put-price').textContent = data.put.toFixed(4);
            document.getElementById('results').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error calculating prices: ' + error);
    }
});

function displayCalibratedParams(params) {
    const paramsDiv = document.getElementById('calibrated-params');
    let html = `
        <p>r: ${params.r.toFixed(4)}</p>
        <p>q: ${params.q.toFixed(4)}</p>
    `;

    if (params.model === 'heston') {
        html += `
            <p>v0: ${params.v0.toFixed(4)}</p>
            <p>kappa: ${params.kappa.toFixed(4)}</p>
            <p>theta: ${params.theta.toFixed(4)}</p>
            <p>sigma: ${params.sigma.toFixed(4)}</p>
            <p>rho: ${params.rho.toFixed(4)}</p>
        `;
    } else if (params.model === 'heston-bates') {
        html += `
            <p>v0: ${params.v0.toFixed(4)}</p>
            <p>kappa: ${params.kappa.toFixed(4)}</p>
            <p>theta: ${params.theta.toFixed(4)}</p>
            <p>sigma: ${params.sigma.toFixed(4)}</p>
            <p>rho: ${params.rho.toFixed(4)}</p>
            <p>lambda: ${params.lambda.toFixed(4)}</p>
            <p>muJ: ${params.muJ.toFixed(4)}</p>
            <p>deltaJ: ${params.deltaJ.toFixed(4)}</p>
        `;
    } else if (params.model === 'double-heston') {
        html += `
            <p>v01: ${params.v01.toFixed(4)}</p>
            <p>kappa1: ${params.kappa1.toFixed(4)}</p>
            <p>theta1: ${params.theta1.toFixed(4)}</p>
            <p>sigma1: ${params.sigma1.toFixed(4)}</p>
            <p>rho1: ${params.rho1.toFixed(4)}</p>
            <p>v02: ${params.v02.toFixed(4)}</p>
            <p>kappa2: ${params.kappa2.toFixed(4)}</p>
            <p>theta2: ${params.theta2.toFixed(4)}</p>
            <p>sigma2: ${params.sigma2.toFixed(4)}</p>
            <p>rho2: ${params.rho2.toFixed(4)}</p>
        `;
    }
    
    paramsDiv.innerHTML = html;
}

function showSheetSelector(sheets, file, model) {
    const selector = document.createElement('div');
    selector.className = 'sheet-selector';
    selector.innerHTML = `
        <h3>Select Sheet</h3>
        <select id="sheet-select">
            ${sheets.map(sheet => `<option value="${sheet}">${sheet}</option>`).join('')}
        </select>
        <button onclick="selectSheet()">Confirm</button>
    `;
    
    document.querySelector('.upload-section').appendChild(selector);
}

async function selectSheet() {
    const fileInput = document.getElementById('excel-file');
    const sheetSelect = document.getElementById('sheet-select');
    const modelSelect = document.getElementById('model');
    const loader = document.getElementById('calibration-loader');
    const formData = new FormData();
    
    formData.append('file', fileInput.files[0]);
    formData.append('sheet_name', sheetSelect.value);
    formData.append('model', modelSelect.value);
    
    try {
        loader.style.display = 'block'; // Show loader
        
        const response = await fetch('/select-sheet', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (response.ok) {
            calibratedParams = data;
            displayCalibratedParams(data);
            document.querySelector('.parameters-section').style.display = 'block';
            document.querySelector('.pricing-section').style.display = 'block';
            document.querySelector('.sheet-selector').remove();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error selecting sheet: ' + error);
    } finally {
        loader.style.display = 'none'; // Hide loader
    }
}

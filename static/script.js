let calibratedParams = null;

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('excel-file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (response.ok) {
            calibratedParams = data;
            displayCalibratedParams(data);
            document.querySelector('.parameters-section').style.display = 'block';
            document.querySelector('.pricing-section').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error uploading file: ' + error);
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
    paramsDiv.innerHTML = `
        <p>v0: ${params.v0.toFixed(4)}</p>
        <p>kappa: ${params.kappa.toFixed(4)}</p>
        <p>theta: ${params.theta.toFixed(4)}</p>
        <p>sigma: ${params.sigma.toFixed(4)}</p>
        <p>rho: ${params.rho.toFixed(4)}</p>
        <p>r: ${params.r.toFixed(4)}</p>
        <p>q: ${params.q.toFixed(4)}</p>
    `;
}

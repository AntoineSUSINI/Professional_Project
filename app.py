from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

app = Flask(__name__)

def heston_charfunc(u, params):
    S0   = params["S0"]
    K    = params["K"]
    r    = params["r"]
    q    = params["q"]
    v0   = params["v0"]
    kappa= params["kappa"]
    theta= params["theta"]
    sigma= params["sigma"]
    rho  = params["rho"]
    T    = params["T"]

    iu = 1j * u

    alpha = -0.5 * (u**2 + iu)
    beta  = kappa - rho * sigma * iu
    gamma = 0.5 * sigma**2

    d = np.sqrt(beta**2 - 4.0 * alpha * gamma)
    g = (beta - d) / (beta + d)

    exp_dT = np.exp(-d * T)

    C = (r - q) * iu * T + \
        (kappa * theta / gamma) * ((beta - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))

    D = (beta - d) / gamma * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + iu * np.log(S0 * np.exp(-q * T)))

def heston_price_call_fft(params, N=10000, U_max=1000):
    S0 = params["S0"]
    r  = params["r"]
    q  = params["q"]
    T  = params["T"]
    K  = params["K"]

    if N % 2 == 1:
        N += 1

    u = np.linspace(1e-10, U_max, N + 1)  # avoid u=0 singularity
    du = u[1] - u[0]
    lnK = np.log(K)

    phi_u     = heston_charfunc(u, params)
    phi_u_im1 = heston_charfunc(u - 1j, params)
    phi_im1   = heston_charfunc(-1j, params)

    integrand_P1 = np.real(np.exp(-1j * u * lnK) * phi_u_im1 /
                           (1j * u * phi_im1))
    integrand_P2 = np.real(np.exp(-1j * u * lnK) * phi_u /
                           (1j * u))

    weights = np.ones(N + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    P1 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand_P1)
    P2 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand_P2)

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    put = K*np.exp(-r*T)*(1-P2) - S0*np.exp(-q*T)*(1-P1)
    return call, put

def bs_call_price(S, K, r, q, sigma, T):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def pack_params(x, S0, r, q):
    return {
        "S0": S0,
        "r": r,
        "q": q,
        "v0": x[0],
        "kappa": x[1],
        "theta": x[2],
        "sigma": x[3],
        "rho": x[4],
    }

@app.route('/')
def index():
    return render_template('index.html')

def get_sheet_name(file):
    """Retourne le nom de la feuille à utiliser ou un dictionnaire avec la liste des feuilles"""
    xls = pd.ExcelFile(file)
    sheet_names = xls.sheet_names
    
    if len(sheet_names) == 1:
        return sheet_names[0]
    
    return {
        'multiple_sheets': True,
        'sheets': sheet_names
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Vérifier les feuilles disponibles
        sheet_result = get_sheet_name(file)
        
        # Si c'est un dictionnaire, plusieurs feuilles sont disponibles
        if isinstance(sheet_result, dict):
            return jsonify(sheet_result)
        
        # Sinon, c'est le nom de l'unique feuille
        wb = pd.read_excel(file, header=None, sheet_name=sheet_result)
        
        # Extraction des données importantes
        S0 = float(wb.iat[3, 4])
        r = float(wb.iat[1, 10])
        q = float(wb.iat[2, 10])
        
        # Extraction de la surface de volatilité
        market_data = extract_market_data(wb)
        
        # Calibration du modèle
        x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5])
        lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999]
        ub = [2.0, 10.0, 2.0, 5.0, 0.999]
        
        opt = least_squares(
            lambda x: residuals(x, market_data, S0, r, q),
            x0,
            bounds=(lb, ub),
            xtol=1e-6,
            ftol=1e-6,
        )
        
        # Stockage des paramètres calibrés
        calibrated_params = {
            'v0': float(opt.x[0]),
            'kappa': float(opt.x[1]),
            'theta': float(opt.x[2]),
            'sigma': float(opt.x[3]),
            'rho': float(opt.x[4]),
            'S0': S0,
            'r': r,
            'q': q
        }
        
        return jsonify(calibrated_params)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select-sheet', methods=['POST'])
def select_sheet():
    if 'file' not in request.files or 'sheet_name' not in request.form:
        return jsonify({'error': 'Missing file or sheet name'}), 400
    
    file = request.files['file']
    sheet_name = request.form['sheet_name']
    
    try:
        # Lecture du fichier Excel avec la feuille sélectionnée
        wb = pd.read_excel(file, header=None, sheet_name=sheet_name)
        
        # Extraction des données importantes
        S0 = float(wb.iat[3, 4])
        r = float(wb.iat[1, 10])
        q = float(wb.iat[2, 10])
        
        # Extraction de la surface de volatilité
        market_data = extract_market_data(wb)
        
        # Calibration du modèle
        x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5])
        lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999]
        ub = [2.0, 10.0, 2.0, 5.0, 0.999]
        
        opt = least_squares(
            lambda x: residuals(x, market_data, S0, r, q),
            x0,
            bounds=(lb, ub),
            xtol=1e-6,
            ftol=1e-6,
        )
        
        # Stockage des paramètres calibrés
        calibrated_params = {
            'v0': float(opt.x[0]),
            'kappa': float(opt.x[1]),
            'theta': float(opt.x[2]),
            'sigma': float(opt.x[3]),
            'rho': float(opt.x[4]),
            'S0': S0,
            'r': r,
            'q': q
        }
        
        return jsonify(calibrated_params)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/price', methods=['POST'])
def calculate_price():
    data = request.json
    
    params = {
        'S0': float(data['S0']),
        'K': float(data['K']),
        'T': float(data['T']),
        'r': float(data['r']),
        'q': float(data['q']),
        'v0': float(data['v0']),
        'kappa': float(data['kappa']),
        'theta': float(data['theta']),
        'sigma': float(data['sigma']),
        'rho': float(data['rho'])
    }
    
    try:
        call, put = heston_price_call_fft(params)
        return jsonify({
            'call': float(call),
            'put': float(put)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_market_data(wb):
    header_row_idx = 6
    start_col = 2
    end_col = 13
    n_maturities = 14

    raw_header = wb.iloc[header_row_idx, start_col:end_col].tolist()
    raw_data   = wb.iloc[
        header_row_idx+1 : header_row_idx+1+n_maturities,
        start_col : end_col
    ]

    df = pd.DataFrame(
        raw_data.values[:, 1:],
        index=raw_data.values[:, 0],
        columns=raw_header[1:],
    )
    df.columns = [f"{x:.1%}" for x in df.columns.astype(float)]

    S0 = wb.iat[3, 4]

    # Extract cell K2
    r = wb.iat[1, 10]

    # Extract cell K3
    q = wb.iat[2, 10]

    strikes_abs = df.loc['T']
    market_data = []
    for mat in df.index:
        T = maturity_to_T(mat)
        if T is None:
            continue
        for pct in df.columns:
            K = strikes_abs[pct]
            vol_imp = df.loc[mat, pct] / 100.0
            price = bs_call_price(S0, K, r, q, vol_imp, T)
            market_data.append((T, K, price))

    return np.array(market_data, dtype=[('T', float), ('K', float), ('price', float)])

def maturity_to_T(label):
    if label == 'T':
        return None
    if label.endswith('M'):
        return int(label[:-1]) / 12.0
    if label.endswith('Y'):
        return float(label[:-1])
    raise ValueError(label)

def residuals(x, market_data, S0, r, q):
    params_base = pack_params(x, S0, r, q)
    res = []
    for T, K, mkt_price in market_data:
        params = params_base.copy()
        params.update({"T": T, "K": K})
        model_call, _ = heston_price_call_fft(params, N=1000, U_max=2000)
        res.append(model_call - mkt_price)
    return np.array(res)

if __name__ == '__main__':
    app.run(debug=True)
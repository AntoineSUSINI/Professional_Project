from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

app = Flask(__name__)

def heston_charfunc(u, params):
    S0   = params["S0"]
    r    = params["r"]
    q    = params["q"]
    T    = params["T"]
    v0   = params["v0"]
    kappa= params["kappa"]
    theta= params["theta"]
    sigma= params["sigma"]
    rho  = params["rho"]

    iu = 1j * u
    b  = kappa - rho * sigma * iu
    d  = np.sqrt(b**2 + sigma**2 * (u**2 + iu))
    g  = (b - d) / (b + d)

    exp_dT = np.exp(-d * T)

    C = iu * (r - q) * T \
        + (kappa * theta / sigma**2) * ((b - d) * T
           - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))

    D = (b - d) / sigma**2 * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + iu * np.log(S0))

def heston_bates_charfunc(u, params):
    S0    = params["S0"]
    r     = params["r"]
    q     = params["q"]
    lam     = params["lambda"]
    muJ    = params["muJ"]
    deltaJ    = params["deltaJ"]
    v0    = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho   = params["rho"]
    T     = params["T"]

    iu = 1j * u

    b  = kappa - rho * sigma * iu
    d  = np.sqrt(b**2 + sigma**2 * (u**2 + iu))
    g  = (b - d) / (b + d)
    e_dT = np.exp(-d * T)

    C = iu * (r - q) * T \
        + (kappa * theta / sigma**2) * (
            (b - d) * T - 2.0 * np.log((1.0 - g * e_dT) / (1.0 - g))
        )
    D = (b - d) / sigma**2 * ((1.0 - e_dT) / (1.0 - g * e_dT))


    jump_cf = np.exp(iu * muJ - 0.5 * deltaJ**2 * u**2)   # φ_J(u)
    C += lam * T * (jump_cf - 1.0)                        # λT(φ_J(u)‑1)


    return np.exp(C + D * v0 + iu * np.log(S0))

def double_heston_charfunc(u, params):
    S0 = params["S0"]
    r  = params["r"]
    q  = params["q"]
    T  = params["T"]

    # Parameters of variance process 1
    v01    = params["v01"]
    kappa1 = params["kappa1"]
    theta1 = params["theta1"]
    sigma1 = params["sigma1"]
    rho1   = params["rho1"]

    # Parameters of variance process 2
    v02    = params["v02"]
    kappa2 = params["kappa2"]
    theta2 = params["theta2"]
    sigma2 = params["sigma2"]
    rho2   = params["rho2"]

    iu = 1j * u

    # --- Fonction auxiliaire pour un facteur « Heston‑like » ---
    def _C_D(kappa, theta, sigma, rho):
        b = kappa - rho * sigma * iu
        d = np.sqrt(b**2 + sigma**2 * (u**2 + iu))
        g = (b - d) / (b + d)
        e_dT = np.exp(-d * T)

        C = (kappa * theta / sigma**2) * (
            (b - d) * T - 2.0 * np.log((1.0 - g * e_dT) / (1.0 - g))
        )
        D = (b - d) / sigma**2 * ((1.0 - e_dT) / (1.0 - g * e_dT))
        return C, D

    C1, D1 = _C_D(kappa1, theta1, sigma1, rho1)
    C2, D2 = _C_D(kappa2, theta2, sigma2, rho2)

    C_total = iu * (r - q) * T + C1 + C2

    # φ(u) complet
    return np.exp(C_total + D1 * v01 + D2 * v02 + iu * np.log(S0))

def heston_price_call(params, N=10000, U_max=1000):
    S0 = params["S0"]
    r  = params["r"]
    q  = params["q"]
    T  = params["T"]
    K  = params["K"]

    # règle de Simpson : N doit être pair
    if N % 2 == 1:
        N += 1

    u   = np.linspace(1e-10, U_max, N + 1)      # éviter u=0
    du  = u[1] - u[0]
    lnK = np.log(K)

    phi_u     = heston_charfunc(u, params)
    phi_u_im1 = heston_charfunc(u - 1j, params)
    phi_im1   = heston_charfunc(-1j, params)

    integrand_P1 = np.real(
        np.exp(-1j * u * lnK) * phi_u_im1 / (1j * u * phi_im1)
    )
    integrand_P2 = np.real(
        np.exp(-1j * u * lnK) * phi_u / (1j * u)
    )

    # poids de Simpson (1,4,2,4,2,...,4,1)
    weights = np.ones(N + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    P1 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand_P1)
    P2 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand_P2)

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    put  = K * np.exp(-r * T) * (1.0 - P2) - S0 * np.exp(-q * T) * (1.0 - P1)

    return call, put


def heston_price_call_bates(params, N=10000, U_max=1000):
    S0 = params["S0"]
    r  = params["r"]
    q  = params["q"]
    T  = params["T"]
    K  = params["K"]

    # Simpson : exiger N pair
    if N % 2 == 1:
        N += 1

    u  = np.linspace(1e-10, U_max, N + 1)   # éviter u = 0
    du = u[1] - u[0]
    lnK = np.log(K)

    phi_u      = heston_bates_charfunc(u, params)
    phi_u_im1  = heston_bates_charfunc(u - 1j, params)
    phi_im1    = heston_bates_charfunc(-1j, params)

    integrand_P1 = np.real(
        np.exp(-1j * u * lnK) * phi_u_im1 / (1j * u * phi_im1)
    )
    integrand_P2 = np.real(
        np.exp(-1j * u * lnK) * phi_u / (1j * u)
    )

    # poids de Simpson (1,4,2,4,2,...,4,1)
    weights = np.ones(N + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    P1 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand_P1)
    P2 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand_P2)

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    put  = K * np.exp(-r * T) * (1.0 - P2) - S0 * np.exp(-q * T) * (1.0 - P1)

    return call, put

def heston_price_call_double_heston(params, N=10000, U_max=1000):
    S0 = params["S0"]
    r  = params["r"]
    q  = params["q"]
    T  = params["T"]
    K  = params["K"]

    # Simpson -> N pair
    if N % 2 == 1:
        N += 1

    u  = np.linspace(1e-10, U_max, N + 1)
    du = u[1] - u[0]
    lnK = np.log(K)

    φ_u      = double_heston_charfunc(u, params)
    φ_u_im1  = double_heston_charfunc(u - 1j, params)
    φ_im1    = double_heston_charfunc(-1j, params)

    I1 = np.real(np.exp(-1j * u * lnK) * φ_u_im1 / (1j * u * φ_im1))
    I2 = np.real(np.exp(-1j * u * lnK) * φ_u / (1j * u))

    w = np.ones(N + 1)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0

    P1 = 0.5 + (du / (3 * np.pi)) * np.sum(w * I1)
    P2 = 0.5 + (du / (3 * np.pi)) * np.sum(w * I2)

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    put  = K * np.exp(-r * T) * (1.0 - P2) - S0 * np.exp(-q * T) * (1.0 - P1)
    return call, put

def heston_price_binary(S0, K, T, params, N=10000, U_max=1000):
    """
    Calcule le prix d'une option binaire (cash-or-nothing) call/put sous Heston.
    """
    params = params.copy()
    params["S0"] = S0
    params["K"] = K
    params["T"] = T
    r = params["r"]

    if N % 2 == 1:
        N += 1

    u = np.linspace(1e-10, U_max, N + 1)
    du = u[1] - u[0]
    lnK = np.log(K)

    phi_u = heston_charfunc(u, params)
    integrand = np.real(np.exp(-1j * u * lnK) * phi_u / (1j * u))

    weights = np.ones(N + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    P2 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand)

    binary_call = np.exp(-r * T) * P2
    binary_put = np.exp(-r * T) * (1 - P2)
    return binary_call, binary_put

def heston_bates_price_binary(S0, K, T, params, N=10000, U_max=1000):
    """
    Calcule le prix d'une option binaire (cash-or-nothing) call/put sous Heston-Bates.
    """
    params = params.copy()
    params["S0"] = S0
    params["K"] = K
    params["T"] = T
    r = params["r"]

    if N % 2 == 1:
        N += 1

    u = np.linspace(1e-10, U_max, N + 1)
    du = u[1] - u[0]
    lnK = np.log(K)

    phi_u = heston_bates_charfunc(u, params)
    integrand = np.real(np.exp(-1j * u * lnK) * phi_u / (1j * u))

    weights = np.ones(N + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    P2 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand)

    binary_call = np.exp(-r * T) * P2
    binary_put = np.exp(-r * T) * (1 - P2)
    return binary_call, binary_put

def double_heston_price_binary(S0, K, T, params, N=10000, U_max=1000):
    """
    Calcule le prix d'une option binaire (cash-or-nothing) call/put sous Double Heston.
    """
    params = params.copy()
    params["S0"] = S0
    params["K"] = K
    params["T"] = T
    r = params["r"]

    if N % 2 == 1:
        N += 1

    u = np.linspace(1e-10, U_max, N + 1)
    du = u[1] - u[0]
    lnK = np.log(K)

    phi_u = double_heston_charfunc(u, params)
    integrand = np.real(np.exp(-1j * u * lnK) * phi_u / (1j * u))

    weights = np.ones(N + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    P2 = 0.5 + (du / (3.0 * np.pi)) * np.sum(weights * integrand)

    binary_call = np.exp(-r * T) * P2
    binary_put = np.exp(-r * T) * (1 - P2)
    return binary_call, binary_put

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

def pack_params_bates(x, S0, r, q):
    return {
        "S0": S0,
        "r": r,
        "q": q,
        "v0": x[0],
        "kappa": x[1],
        "theta": x[2],
        "sigma": x[3],
        "rho": x[4],
        "lambda": x[5],
        "muJ": x[6],
        "deltaJ": x[7]
    }

def pack_params_double_heston(x, S0, r, q):
    return {
        "S0": S0,
        "r": r,
        "q": q,
        "v01": x[0],
        "kappa1": x[1],
        "theta1": x[2],
        "sigma1": x[3],
        "rho1": x[4],
        "v02": x[5],
        "kappa2": x[6],
        "theta2": x[7],
        "sigma2": x[8],
        "rho2": x[9],
    }

@app.route('/')
def index():
    return render_template('index.html')

def get_sheet_name(file):
    """Selection de la feuille dans le fichier Excel"""
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
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No file uploaded or model not specified'}), 400
    
    file = request.files['file']
    model = request.form['model']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        sheet_result = get_sheet_name(file)
        
        if isinstance(sheet_result, dict):
            return jsonify(sheet_result)
        
        wb = pd.read_excel(file, header=None, sheet_name=sheet_result)
        
        S0 = float(wb.iat[3, 4])
        r = float(wb.iat[1, 10])
        q = float(wb.iat[2, 10])
        
        market_data = extract_market_data(wb)
        
        if model == 'heston':
            x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5])
            lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999]
            ub = [2.0, 10.0, 2.0, 5.0, 0.999]
            residuals_func = lambda x: residuals(x, market_data, S0, r, q)
        elif model == 'heston-bates':
            x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5, 0.1, -0.02, 0.1])
            lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999, 0.0, -1.0, 1e-4]
            ub = [2.0, 10.0, 2.0, 5.0, 0.999, 5.0, 1.0, 5.0]
            residuals_func = lambda x: residuals_bates(x, market_data, S0, r, q)
        else:  # double-heston
            x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5, 0.04, 1.0, 0.04, 0.5, -0.5])
            lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999, 1e-4, 1e-4, 1e-4, 1e-4, -0.999]
            ub = [2.0, 10.0, 2.0, 5.0, 0.999, 2.0, 10.0, 2.0, 5.0, 0.999]
            residuals_func = lambda x: residuals_double_heston(x, market_data, S0, r, q)
        
        opt = least_squares(
            residuals_func,
            x0,
            bounds=(lb, ub),
            xtol=1e-6,
            ftol=1e-6,
        )
        
        if model == 'heston':
            calibrated_params = {
                'model': 'heston',
                'v0': float(opt.x[0]),
                'kappa': float(opt.x[1]),
                'theta': float(opt.x[2]),
                'sigma': float(opt.x[3]),
                'rho': float(opt.x[4]),
                'S0': S0,
                'r': r,
                'q': q
            }
        elif model == 'heston-bates':
            calibrated_params = {
                'model': 'heston-bates',
                'v0': float(opt.x[0]),
                'kappa': float(opt.x[1]),
                'theta': float(opt.x[2]),
                'sigma': float(opt.x[3]),
                'rho': float(opt.x[4]),
                'lambda': float(opt.x[5]),
                'muJ': float(opt.x[6]),
                'deltaJ': float(opt.x[7]),
                'S0': S0,
                'r': r,
                'q': q
            }
        else:
            calibrated_params = {
                'model': 'double-heston',
                'v01': float(opt.x[0]),
                'kappa1': float(opt.x[1]),
                'theta1': float(opt.x[2]),
                'sigma1': float(opt.x[3]),
                'rho1': float(opt.x[4]),
                'v02': float(opt.x[5]),
                'kappa2': float(opt.x[6]),
                'theta2': float(opt.x[7]),
                'sigma2': float(opt.x[8]),
                'rho2': float(opt.x[9]),
                'S0': S0,
                'r': r,
                'q': q
            }
        
        return jsonify(calibrated_params)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select-sheet', methods=['POST'])
def select_sheet():
    if 'file' not in request.files or 'sheet_name' not in request.form or 'model' not in request.form:
        return jsonify({'error': 'Missing file, sheet name or model'}), 400
    
    file = request.files['file']
    sheet_name = request.form['sheet_name']
    model = request.form['model']
    
    try:
        wb = pd.read_excel(file, header=None, sheet_name=sheet_name)
        
        S0 = float(wb.iat[3, 4])
        r = float(wb.iat[1, 10])
        q = float(wb.iat[2, 10])
        
        market_data = extract_market_data(wb)
        
        if model == 'heston':
            x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5])
            lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999]
            ub = [2.0, 10.0, 2.0, 5.0, 0.999]
            residuals_func = lambda x: residuals(x, market_data, S0, r, q)
        elif model == 'heston-bates':
            x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5, 0.1, -0.02, 0.1])
            lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999, 0.0, -1.0, 1e-4]
            ub = [2.0, 10.0, 2.0, 5.0, 0.999, 5.0, 1.0, 5.0]
            residuals_func = lambda x: residuals_bates(x, market_data, S0, r, q)
        else:  # double-heston
            x0 = np.array([0.04, 1.0, 0.04, 0.5, -0.5, 0.04, 1.0, 0.04, 0.5, -0.5])
            lb = [1e-4, 1e-4, 1e-4, 1e-4, -0.999, 1e-4, 1e-4, 1e-4, 1e-4, -0.999]
            ub = [2.0, 10.0, 2.0, 5.0, 0.999, 2.0, 10.0, 2.0, 5.0, 0.999]
            residuals_func = lambda x: residuals_double_heston(x, market_data, S0, r, q)
        
        opt = least_squares(
            residuals_func,
            x0,
            bounds=(lb, ub),
            xtol=1e-6,
            ftol=1e-6,
        )
        
        if model == 'heston':
            calibrated_params = {
                'model': 'heston',
                'v0': float(opt.x[0]),
                'kappa': float(opt.x[1]),
                'theta': float(opt.x[2]),
                'sigma': float(opt.x[3]),
                'rho': float(opt.x[4]),
                'S0': S0,
                'r': r,
                'q': q
            }
        elif model == 'heston-bates':
            calibrated_params = {
                'model': 'heston-bates',
                'v0': float(opt.x[0]),
                'kappa': float(opt.x[1]),
                'theta': float(opt.x[2]),
                'sigma': float(opt.x[3]),
                'rho': float(opt.x[4]),
                'lambda': float(opt.x[5]),
                'muJ': float(opt.x[6]),
                'deltaJ': float(opt.x[7]),
                'S0': S0,
                'r': r,
                'q': q
            }
        else:
            calibrated_params = {
                'model': 'double-heston',
                'v01': float(opt.x[0]),
                'kappa1': float(opt.x[1]),
                'theta1': float(opt.x[2]),
                'sigma1': float(opt.x[3]),
                'rho1': float(opt.x[4]),
                'v02': float(opt.x[5]),
                'kappa2': float(opt.x[6]),
                'theta2': float(opt.x[7]),
                'sigma2': float(opt.x[8]),
                'rho2': float(opt.x[9]),
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
    model = data.get('model', 'heston')
    option_type = data.get('option_type', 'vanilla')  # 'vanilla' ou 'binary'

    if model == 'heston':
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
        if option_type == 'binary':
            pricing_func = heston_price_binary
        else:
            pricing_func = heston_price_call
    elif model == 'heston-bates':
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
            'rho': float(data['rho']),
            'lambda': float(data['lambda']),
            'muJ': float(data['muJ']),
            'deltaJ': float(data['deltaJ'])
        }
        if option_type == 'binary':
            pricing_func = heston_bates_price_binary
        else:
            pricing_func = heston_price_call_bates
    else:  # double-heston
        params = {
            'S0': float(data['S0']),
            'K': float(data['K']),
            'T': float(data['T']),
            'r': float(data['r']),
            'q': float(data['q']),
            'v01': float(data['v01']),
            'kappa1': float(data['kappa1']),
            'theta1': float(data['theta1']),
            'sigma1': float(data['sigma1']),
            'rho1': float(data['rho1']),
            'v02': float(data['v02']),
            'kappa2': float(data['kappa2']),
            'theta2': float(data['theta2']),
            'sigma2': float(data['sigma2']),
            'rho2': float(data['rho2'])
        }
        if option_type == 'binary':
            pricing_func = double_heston_price_binary
        else:
            pricing_func = heston_price_call_double_heston

    try:
        call, put = pricing_func(
            params['S0'], params['K'], params['T'], params
        ) if option_type == 'binary' else pricing_func(params)
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

    r = wb.iat[1, 10]

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
        model_call, _ = heston_price_call(params, N=1000, U_max=2000)
        res.append(model_call - mkt_price)
    return np.array(res)

def residuals_bates(x, market_data, S0, r, q):
    params_base = pack_params_bates(x, S0, r, q)
    res = []
    for T, K, mkt_price in market_data:
        params = params_base.copy()
        params.update({"T": T, "K": K})
        model_call, _ = heston_price_call_bates(params, N=1000, U_max=2000)
        res.append(model_call - mkt_price)
    return np.array(res)

def residuals_double_heston(x, market_data, S0, r, q):
    params_base = pack_params_double_heston(x, S0, r, q)
    res = []
    for T, K, mkt_price in market_data:
        params = params_base.copy()
        params.update({"T": T, "K": K})
        model_call, _ = heston_price_call_double_heston(params, N=2000, U_max=2000)
        res.append(model_call - mkt_price)
    return np.array(res)

if __name__ == '__main__':
    app.run(debug=True)
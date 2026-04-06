import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Estrutura geral da simulacao:
# - fluxo principal de execucao
# - organizacao modular das funcoes
# - exibicao das metricas de desempenho
# - comparacao entre micro-ondas e FSO
import final as base_final

# Modelo FSO simplificado baseado em orcamento de enlace:
# - modulacao PAM-4 optica
# - perdas geometricas, atmosfericas e de sistema
# - ruido gaussiano aditivo


DEBUG = False

FILE_PATH = base_final.FILE_PATH
MAX_CHARS = base_final.MAX_CHARS

RSYM = base_final.RSYM
BITS_PER_SYMBOL = base_final.BITS_PER_SYMBOL
BIT_RATE = base_final.BIT_RATE

DEFAULT_DISTANCE = base_final.DEFAULT_DISTANCE
DEFAULT_NOISE_POWER = base_final.DEFAULT_NOISE_POWER
NOISE_LEVELS = base_final.NOISE_LEVELS

PROPAGATION_SPEED_MW = base_final.PROPAGATION_SPEED
PROPAGATION_SPEED_FSO = base_final.LIGHT_SPEED
JITTER_STD_FACTOR = base_final.JITTER_STD_FACTOR


TECH_CONFIG = {
    "mw": {
        "label": "Micro-ondas",
        "table_label": "MICROONDAS",
        "frequency_hz": base_final.TECH_CONFIG["mw"]["frequency_hz"],
    },
    "fso": {
        "label": "FSO",
        "table_label": "FSO",
        "wavelength_m": 1550e-9,
    },
}


@dataclass
class SimplifiedFSOParams:
    """
    Parametros do enlace FSO.
    """
    Pt: float = 0.5
    wavelength: float = 1550e-9
    link_distance: float = DEFAULT_DISTANCE
    beam_divergence: float = 2e-3
    rx_aperture_diameter: float = 0.05
    atmospheric_atten_dB_per_km: float = 10.0
    system_loss_db: float = 3.0
    noise_std: float = 1e-5
    random_seed: int = 42


DEFAULT_FSO_PARAMS = SimplifiedFSOParams(
    Pt=0.5,
    wavelength=1550e-9,
    link_distance=DEFAULT_DISTANCE,
    beam_divergence=2e-3,
    rx_aperture_diameter=0.05,
    atmospheric_atten_dB_per_km=10.0,
    system_loss_db=3.0,
    noise_std=1e-5,
    random_seed=42,
)


def debug_print(*args):
    if DEBUG:
        print(*args)


def load_input_text(file_path, max_chars=None):
    return base_final.load_input_text(file_path, max_chars)


def text_to_bits(text):
    return base_final.text_to_bits(text)


def pam4_mod_mw(bits):
    """Modulacao PAM-4 aplicada ao enlace de micro-ondas."""
    return base_final.pam4_mod(bits)


def pam4_demod_mw(symbols):
    return base_final.pam4_demod(symbols)


def pam4_mod_fso(bits):
    """
    Modulacao PAM-4 optica com niveis normalizados de intensidade.

    Mapeia pares de bits para niveis normalizados de intensidade:
    00 -> 0.0, 01 -> 1/3, 10 -> 2/3, 11 -> 1.0
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    pairs = bits.reshape(-1, 2)
    idx = pairs[:, 0] * 2 + pairs[:, 1]
    levels = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    return levels[idx]


def pam4_demod_fso(symbols_rx):
    """Demodulacao PAM-4 optica por decisao de nivel mais proximo."""
    levels = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    idx_hat = np.argmin(np.abs(symbols_rx[:, None] - levels[None, :]), axis=1)

    bits_out = np.zeros(2 * len(idx_hat), dtype=int)
    bits_out[0::2] = idx_hat // 2
    bits_out[1::2] = idx_hat % 2
    return bits_out


def calculate_mw_path_loss(distance):
    """Calcula a perda de percurso do enlace de micro-ondas."""
    return base_final.calculate_path_loss(distance, "mw")


def geometric_loss_db(distance, beam_divergence, rx_aperture_diameter):
    """
    Perda geometrica do FSO.

    O modelo considera o espalhamento do feixe ao longo da distancia:
    - raio do spot: theta * L / 2
    - ganho geometrico: area receptora / area do spot, limitado a 1
    - perda geometrica em dB: -10 log10(ganho)
    """
    spot_radius = beam_divergence * distance / 2.0
    if spot_radius <= 0:
        return 0.0

    spot_area = np.pi * spot_radius ** 2
    rx_area = np.pi * (rx_aperture_diameter / 2.0) ** 2
    geometric_gain = min(rx_area / spot_area, 1.0)
    return float(-10.0 * np.log10(geometric_gain + 1e-30))


def atmospheric_loss_db(distance, atmospheric_atten_dB_per_km):
    """
    Perda atmosferica simplificada:
    atenuacao especifica [dB/km] x distancia [km].
    """
    return float(atmospheric_atten_dB_per_km * (distance / 1000.0))


def build_fso_params(distance, noise_power, n_bits):
    """
    Define os parametros do enlace FSO para a simulacao atual.

    O modelo considera:
    - perda geometrica
    - perda atmosferica
    - perda fixa de sistema
    - ruido gaussiano simples
    """
    noise_std = float(np.sqrt(max(noise_power, 1e-30)))

    return SimplifiedFSOParams(
        Pt=DEFAULT_FSO_PARAMS.Pt,
        wavelength=DEFAULT_FSO_PARAMS.wavelength,
        link_distance=distance,
        beam_divergence=DEFAULT_FSO_PARAMS.beam_divergence,
        rx_aperture_diameter=DEFAULT_FSO_PARAMS.rx_aperture_diameter,
        atmospheric_atten_dB_per_km=DEFAULT_FSO_PARAMS.atmospheric_atten_dB_per_km,
        system_loss_db=DEFAULT_FSO_PARAMS.system_loss_db,
        noise_std=noise_std,
        random_seed=DEFAULT_FSO_PARAMS.random_seed,
    )


def generate_delay(num_symbols, distance, propagation_speed):
    """
    Gera o atraso de propagacao e o jitter simbolo a simbolo.

    O atraso medio depende da distancia e da velocidade de propagacao,
    enquanto o jitter e representado por uma perturbacao gaussiana.
    """
    symbol_period = 1 / RSYM
    tx_time = np.arange(num_symbols) * symbol_period
    base_delay = distance / propagation_speed
    jitter_process = np.random.randn(num_symbols) * symbol_period * JITTER_STD_FACTOR
    rx_time = tx_time + base_delay + jitter_process
    return rx_time - tx_time


def channel_mw(symbols, noise_power, distance, los=True):
    """Simula o canal de micro-ondas."""
    return base_final.channel(symbols, noise_power, distance, tech="mw", los=los)


def channel_fso(symbols, noise_power, distance, rng=None):
    """
    Canal FSO simplificado.

    O modelo e baseado em orcamento de enlace e considera:
    - A_geo: perda geometrica
    - A_atm: perda atmosferica
    - A_system: perda fixa de sistema
    - ruido gaussiano simples.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_FSO_PARAMS.random_seed)

    params = build_fso_params(
        distance,
        noise_power,
        n_bits=len(symbols) * BITS_PER_SYMBOL,
    )
    n_symbols = len(symbols)

    geometric_loss = geometric_loss_db(
        params.link_distance,
        params.beam_divergence,
        params.rx_aperture_diameter,
    )
    atmospheric_loss = atmospheric_loss_db(
        params.link_distance,
        params.atmospheric_atten_dB_per_km,
    )
    system_loss = params.system_loss_db
    total_loss_db = geometric_loss + atmospheric_loss + system_loss
    total_gain_linear = 10 ** (-total_loss_db / 10.0)

    optical_power_tx = params.Pt * symbols
    optical_power_rx = optical_power_tx * total_gain_linear

    noise = rng.normal(0.0, params.noise_std, size=n_symbols)
    received_signal = optical_power_rx + noise

    equalized_signal = received_signal / (params.Pt * total_gain_linear + 1e-30)
    equalized_signal = np.clip(equalized_signal, 0.0, 1.0)

    delay = generate_delay(n_symbols, distance, PROPAGATION_SPEED_FSO)

    return {
        "received_signal": received_signal,
        "equalized_signal": equalized_signal,
        "delay": delay,
        "path_loss_db": float(total_loss_db),
        "geometric_loss_db": float(geometric_loss),
        "atmospheric_loss_db": float(atmospheric_loss),
        "system_loss_db": float(system_loss),
        "total_fso_loss_db": float(total_loss_db),
        "total_gain_linear": float(total_gain_linear),
        "mean_received_power_W": float(np.mean(optical_power_rx)),
    }


def calculate_metrics(bits_tx, bits_rx, delay, bit_rate):
    return base_final.calculate_metrics(bits_tx, bits_rx, delay, bit_rate)


def simulate_link(bits_tx, tech, distance, noise_power, los=True, seed=None):
    """
    Executa a simulacao de um enlace.

    Cada tecnologia utiliza sua propria modulacao e seu respectivo modelo
    de canal:
    - MW: PAM-4 eletrico e canal de radiofrequencia
    - FSO: PAM-4 optico + canal simplificado por perdas
    """
    tech_config = TECH_CONFIG[tech]
    rng = np.random.default_rng(seed if seed is not None else DEFAULT_FSO_PARAMS.random_seed)

    if tech == "mw":
        symbols_tx = pam4_mod_mw(bits_tx)
        channel_result = channel_mw(symbols_tx, noise_power, distance, los=los)
        equalized_signal = (
            channel_result["received_signal"] / channel_result["attenuation_linear_amplitude"]
        )
        bits_rx = pam4_demod_mw(equalized_signal)[:len(bits_tx)]
        metrics = calculate_metrics(bits_tx, bits_rx, channel_result["delay"], BIT_RATE)

        return {
            "tech": tech,
            "label": tech_config["label"],
            "table_label": tech_config["table_label"],
            "distance": distance,
            "noise_power": noise_power,
            "path_loss_db": channel_result["path_loss_db"],
            "bits_rx": bits_rx,
            "symbols_tx": symbols_tx,
            "received_signal": channel_result["received_signal"],
            "equalized_signal": equalized_signal,
            **metrics,
        }

    if tech == "fso":
        symbols_tx = pam4_mod_fso(bits_tx)
        channel_result = channel_fso(symbols_tx, noise_power, distance, rng=rng)
        bits_rx = pam4_demod_fso(channel_result["equalized_signal"])[:len(bits_tx)]
        metrics = calculate_metrics(bits_tx, bits_rx, channel_result["delay"], BIT_RATE)

        return {
            "tech": tech,
            "label": tech_config["label"],
            "table_label": tech_config["table_label"],
            "distance": distance,
            "noise_power": noise_power,
            "path_loss_db": channel_result["path_loss_db"],
            "bits_rx": bits_rx,
            "symbols_tx": symbols_tx,
            "received_signal": channel_result["received_signal"],
            "equalized_signal": channel_result["equalized_signal"],
            "geometric_loss_db": channel_result["geometric_loss_db"],
            "atmospheric_loss_db": channel_result["atmospheric_loss_db"],
            "system_loss_db": channel_result["system_loss_db"],
            "total_fso_loss_db": channel_result["total_fso_loss_db"],
            "total_gain_linear": channel_result["total_gain_linear"],
            "mean_received_power_W": channel_result["mean_received_power_W"],
            **metrics,
        }

    raise ValueError("tech deve ser 'mw' ou 'fso'")


def simulate_all_links(bits_tx, distance, noise_power, techs=None):
    if techs is None:
        techs = list(TECH_CONFIG.keys())

    return [
        simulate_link(bits_tx, tech, distance, noise_power, seed=DEFAULT_FSO_PARAMS.random_seed)
        for tech in techs
    ]


def choose_best_link(results):
    return base_final.choose_best_link(results)


def print_section(title):
    base_final.print_section(title)


def print_simulation_parameters(text, bits_tx, distance, noise_power):
    print_section("Parametros da Simulacao")
    print(f"Arquivo de entrada : {FILE_PATH}")
    print(f"Caracteres usados  : {len(text)}")
    print(f"Bits transmitidos  : {len(bits_tx)}")
    print(f"Simbolos MW PAM-4  : {len(pam4_mod_mw(bits_tx))}")
    print(f"Simbolos FSO PAM-4 : {len(pam4_mod_fso(bits_tx))}")
    print(f"Taxa bruta         : {BIT_RATE:.2f} bps")
    print(f"Distancia          : {distance:.0f} m")
    print(f"Potencia de ruido  : {noise_power:.2e}")
    print(f"MW                 : {TECH_CONFIG['mw']['frequency_hz'] / 1e9:.2f} GHz")
    print(f"FSO                : {TECH_CONFIG['fso']['wavelength_m'] * 1e9:.0f} nm")
    print("Modelo FSO         : simplificado por perdas (academico)")

    debug_print("Texto (primeiros 200 caracteres):")
    debug_print(text[:200])


def print_main_results(results):
    print_section("Resultado Principal")
    print(
        f"{'Tecnologia':<18} {'Perda (dB)':>12} {'BER':>10} {'Delay (us)':>12} "
        f"{'Jitter (us)':>12} {'Bit Rate':>12} {'Throughput':>12}"
    )

    for result in results:
        print(
            f"{result['label']:<18} {result['path_loss_db']:12.2f} {result['ber']:10.4f} "
            f"{result['delay_mean'] * 1e6:12.2f} {result['jitter'] * 1e6:12.2f} "
            f"{result['bit_rate']:12.2f} {result['throughput']:12.2f}"
        )

    print(f"\nMelhor enlace por throughput: {choose_best_link(results)}")


def print_noise_variation(noise_results):
    print_section("Variacao com Ruido")

    for noise_power, results in noise_results:
        print(f"\nPotencia de ruido: {noise_power:.2e}")
        print(
            f"{'Tecnologia':<18} {'Bits corretos':>15} {'Sucesso (%)':>12} "
            f"{'Bits erro':>12} {'BER':>10} {'Bit Rate':>12} "
            f"{'Throughput':>12} {'Delay (us)':>12} {'Jitter (us)':>12}"
        )

        for result in results:
            print(
                f"{result['table_label']:<18} {result['bits_correct']:15d} "
                f"{result['success_rate'] * 100:12.2f} {result['bits_error']:12d} "
                f"{result['ber']:10.4f} {result['bit_rate']:12.2f} "
                f"{result['throughput']:12.2f} {result['delay_mean'] * 1e6:12.2f} "
                f"{result['jitter'] * 1e6:12.2f}"
            )


def print_fso_details(results):
    print_section("Detalhes do Canal FSO")
    fso_result = next((result for result in results if result["tech"] == "fso"), None)

    if fso_result is None:
        return

    print(f"Perda geometrica [dB]       : {fso_result['geometric_loss_db']:.2f}")
    print(f"Perda atmosferica [dB]      : {fso_result['atmospheric_loss_db']:.2f}")
    print(f"Perda de sistema [dB]       : {fso_result['system_loss_db']:.2f}")
    print(f"Perda total FSO [dB]        : {fso_result['total_fso_loss_db']:.2f}")
    print(f"Potencia media recebida [W] : {fso_result['mean_received_power_W']:.6e}")


def estimate_fso_equivalent_loss(distance, noise_power, seed=DEFAULT_FSO_PARAMS.random_seed):
    """
    Estima a perda total equivalente do enlace FSO.
    """
    params = build_fso_params(distance, noise_power, n_bits=2000)
    geometric_loss = geometric_loss_db(
        params.link_distance,
        params.beam_divergence,
        params.rx_aperture_diameter,
    )
    atmospheric_loss = atmospheric_loss_db(
        params.link_distance,
        params.atmospheric_atten_dB_per_km,
    )
    return geometric_loss + atmospheric_loss + params.system_loss_db


def plot_path_loss(distance_start=100, distance_end=3000, num_points=60):
    """
    Gera um grafico comparativo da perda de enlace em funcao da distancia.

    Para micro-ondas, utiliza-se a perda de espaco livre.
    Para FSO, utiliza-se a perda total equivalente do modelo simplificado.
    """
    distances = np.linspace(distance_start, distance_end, num_points)
    loss_mw = [calculate_mw_path_loss(distance) for distance in distances]
    loss_fso = [
        estimate_fso_equivalent_loss(distance, DEFAULT_NOISE_POWER)
        for distance in distances
    ]

    plt.figure()
    plt.plot(distances, loss_mw, label="Micro-ondas (10 GHz)", color="blue")
    plt.plot(distances, loss_fso, label="FSO (modelo simplificado)", color="red")
    plt.title("Perda do Enlace vs Distancia")
    plt.xlabel("Distancia (m)")
    plt.ylabel("Perda / perda equivalente (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Organiza a execucao da simulacao comparativa.

    O programa compara enlaces de micro-ondas e FSO com modulacao PAM-4,
    calculando BER, atraso medio, jitter, taxa de bits e throughput.
    """
    input_text = load_input_text(FILE_PATH, MAX_CHARS)
    bits_tx = text_to_bits(input_text)

    print_simulation_parameters(
        input_text,
        bits_tx,
        DEFAULT_DISTANCE,
        DEFAULT_NOISE_POWER,
    )

    main_results = simulate_all_links(
        bits_tx,
        DEFAULT_DISTANCE,
        DEFAULT_NOISE_POWER,
    )
    print_main_results(main_results)
    print_fso_details(main_results)

    noise_results = [
        (
            noise_power,
            simulate_all_links(bits_tx, DEFAULT_DISTANCE, noise_power),
        )
        for noise_power in NOISE_LEVELS
    ]
    print_noise_variation(noise_results)

    plot_path_loss()


if __name__ == "__main__":
    main()

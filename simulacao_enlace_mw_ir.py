import numpy as np
import matplotlib.pyplot as plt


DEBUG = False

FILE_PATH = r"D:\UFABC\INF111 - Redes sem fio\code\Frankenstein.txt"
MAX_CHARS = 50000

RSYM = 8000
BITS_PER_SYMBOL = 2
BIT_RATE = RSYM * BITS_PER_SYMBOL

DEFAULT_DISTANCE = 3000
DEFAULT_NOISE_POWER = 1e-10
NOISE_LEVELS = [1e-6, 1e-8, 1e-10, 1e-12]

PROPAGATION_SPEED = 2e8
LIGHT_SPEED = 3e8
TX_GAIN_DB = 30
RX_GAIN_DB = 30
JITTER_STD_FACTOR = 0.05

TECH_CONFIG = {
    "mw": {"label": "Micro-ondas", "table_label": "MICROONDAS", "frequency_hz": 10e9},
    "ir": {"label": "Infravermelho", "table_label": "INFRAVERMELHO", "frequency_hz": 200e12},
}


def debug_print(*args):
    """Exibe mensagens auxiliares apenas quando DEBUG estiver habilitado."""
    if DEBUG:
        print(*args)


def load_input_text(file_path, max_chars=None):
    """Lê o arquivo de entrada e opcionalmente limita a quantidade de caracteres."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    if max_chars is not None:
        text = text[:max_chars]

    return text


def text_to_bits(text):
    """Converte cada caractere do texto em sua representação binária de 8 bits."""
    bits = []
    for char in text:
        bits.extend([int(bit) for bit in format(ord(char), "08b")])
    return np.array(bits)


def pam4_mod(bits):
    """Agrupa os bits de 2 em 2 e os mapeia para símbolos PAM-4."""
    mapping = [-3, -1, 1, 3]
    symbols = []

    for index in range(0, len(bits), 2):
        pair = bits[index:index + 2]

        # Se sobrar 1 bit no final, completa com zero para formar um símbolo.
        if len(pair) < 2:
            pair = np.append(pair, 0)

        mapping_index = pair[0] * 2 + pair[1]
        symbols.append(mapping[mapping_index])

    return np.array(symbols)


def pam4_demod(symbols):
    """Converte símbolos recebidos em bits escolhendo o nível PAM-4 mais próximo."""
    valid_levels = np.array([-3, -1, 1, 3])
    bits = []

    for symbol in symbols:
        # Decide qual símbolo válido está mais perto do valor recebido.
        nearest_index = np.argmin(np.abs(valid_levels - symbol))
        bits.extend([int(bit) for bit in format(nearest_index, "02b")])

    return np.array(bits)


def get_technology_config(tech):
    """Retorna a configuração associada à tecnologia escolhida."""
    if tech not in TECH_CONFIG:
        raise ValueError("tech deve ser 'mw' ou 'ir'")
    return TECH_CONFIG[tech]


def calculate_free_space_path_loss(distance, frequency_hz):
    """Calcula a perda de propagação em espaço livre pela equação de Friis."""
    wavelength = LIGHT_SPEED / frequency_hz
    return 20 * np.log10(4 * np.pi * distance / wavelength)


def calculate_path_loss(distance, tech):
    """Seleciona o cálculo de perda conforme a tecnologia analisada."""
    frequency_hz = get_technology_config(tech)["frequency_hz"]
    return calculate_free_space_path_loss(distance, frequency_hz)


def channel(symbols, noise_power, distance, tech="mw", los=True):
    """
    Simula a transmissão dos símbolos por um enlace sem fio.

    A função combina:
    - perda de percurso com base na equação de Friis;
    - ganho fixo de transmissão e recepção;
    - ruído branco gaussiano com potência fixa;
    - atraso de propagação e jitter temporal;
    - bloqueio total do sinal quando não há linha de visada.
    """
    path_loss_db = calculate_path_loss(distance, tech)
    gain_total_db = TX_GAIN_DB + RX_GAIN_DB

    # Converte a perda líquida do enlace para escala linear e atenua o sinal.
    attenuation_db = -path_loss_db + gain_total_db
    attenuation_linear_power = 10 ** (attenuation_db / 10)
    attenuation_linear_amplitude = np.sqrt(attenuation_linear_power)

    transmitted_signal = symbols * attenuation_linear_amplitude

    # Aplica ruído com a mesma potência fixa para qualquer tecnologia.
    noise = np.sqrt(noise_power) * np.random.randn(len(symbols))
    received_signal = transmitted_signal + noise

    # Gera os instantes de transmissão e recepção de cada símbolo.
    symbol_period = 1 / RSYM
    tx_time = np.arange(len(symbols)) * symbol_period
    base_delay = distance / PROPAGATION_SPEED

    # Modela o jitter como uma perturbação aleatória do tempo ideal de chegada.
    jitter_process = np.random.randn(len(symbols)) * symbol_period * JITTER_STD_FACTOR
    rx_time = tx_time + base_delay + jitter_process
    delay = rx_time - tx_time

    # Sem linha de visada, considera bloqueio completo do enlace.
    if not los:
        received_signal = np.zeros_like(symbols)

    return {
        "received_signal": received_signal,
        "delay": delay,
        "attenuation_linear_amplitude": attenuation_linear_amplitude,
        "path_loss_db": path_loss_db,
    }


def calculate_metrics(bits_tx, bits_rx, delay, bit_rate):
    """Calcula as métricas principais de desempenho do enlace."""
    total_bits = len(bits_tx)
    bits_correct = np.sum(bits_tx == bits_rx)
    bits_error = total_bits - bits_correct
    success_rate = bits_correct / total_bits
    ber = bits_error / total_bits
    throughput = bit_rate * (1 - ber)

    return {
        "total_bits": total_bits,
        "bits_correct": bits_correct,
        "bits_error": bits_error,
        "success_rate": success_rate,
        "ber": ber,
        "delay_mean": np.mean(delay),
        "jitter": np.std(delay),
        "bit_rate": bit_rate,
        "throughput": throughput,
    }


def simulate_link(symbols_tx, bits_tx, tech, distance, noise_power, los=True):
    """Executa a simulação completa de um enlace e retorna suas métricas."""
    tech_config = get_technology_config(tech)
    channel_result = channel(symbols_tx, noise_power, distance, tech=tech, los=los)

    equalized_signal = (
        channel_result["received_signal"] / channel_result["attenuation_linear_amplitude"]
    )
    bits_rx = pam4_demod(equalized_signal)[:len(bits_tx)]
    metrics = calculate_metrics(bits_tx, bits_rx, channel_result["delay"], BIT_RATE)

    return {
        "tech": tech,
        "label": tech_config["label"],
        "table_label": tech_config["table_label"],
        "distance": distance,
        "noise_power": noise_power,
        "path_loss_db": channel_result["path_loss_db"],
        "bits_rx": bits_rx,
        "received_signal": channel_result["received_signal"],
        "equalized_signal": equalized_signal,
        **metrics,
    }


def simulate_all_links(symbols_tx, bits_tx, distance, noise_power, techs=None):
    """Simula todas as tecnologias selecionadas e retorna os resultados."""
    if techs is None:
        techs = list(TECH_CONFIG.keys())

    return [
        simulate_link(symbols_tx, bits_tx, tech, distance, noise_power)
        for tech in techs
    ]


def choose_best_link(results):
    """Seleciona o melhor enlace usando throughput como métrica principal."""
    throughputs = [result["throughput"] for result in results]

    if np.allclose(throughputs, throughputs[0]):
        return "Empate"

    return max(results, key=lambda result: result["throughput"])["label"]


def print_section(title):
    """Exibe um título de seção de forma padronizada."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def print_simulation_parameters(text, bits_tx, symbols_tx, distance, noise_power):
    """Exibe os parâmetros principais usados na simulação."""
    print_section("Parâmetros da Simulação")
    print(f"Arquivo de entrada : {FILE_PATH}")
    print(f"Caracteres usados  : {len(text)}")
    print(f"Bits transmitidos  : {len(bits_tx)}")
    print(f"Símbolos PAM-4     : {len(symbols_tx)}")
    print(f"Taxa bruta         : {BIT_RATE:.2f} bps")
    print(f"Distância          : {distance:.0f} m")
    print(f"Potência de ruído  : {noise_power:.2e}")

    for tech in TECH_CONFIG.values():
        frequency_ghz = tech["frequency_hz"] / 1e9
        print(f"{tech['label']:<18}: {frequency_ghz:.2f} GHz")

    debug_print("Texto (primeiros 200 caracteres):")
    debug_print(text[:200])


def print_main_results(results):
    """Exibe a comparação principal entre as tecnologias."""
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
    """Exibe os resultados para diferentes níveis de potência de ruído."""
    print_section("Variação com Ruído")

    for noise_power, results in noise_results:
        print(f"\nPotência de ruído: {noise_power:.2e}")
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


def plot_path_loss(distance_start=100, distance_end=3000, num_points=100):
    """Gera o gráfico de perda de propagação em função da distância."""
    distances = np.linspace(distance_start, distance_end, num_points)
    loss_mw = [calculate_path_loss(distance, "mw") for distance in distances]
    loss_ir = [calculate_path_loss(distance, "ir") for distance in distances]

    plt.figure()
    plt.plot(distances, loss_mw, label="Micro-ondas (10 GHz)", color="blue")
    plt.plot(distances, loss_ir, label="Infravermelho (200 THz)", color="red")
    plt.title("Perda de Propagação vs Distância (Friis)")
    plt.xlabel("Distância (m)")
    plt.ylabel("Perda (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Organiza a execução da simulação e a apresentação dos resultados."""
    input_text = load_input_text(FILE_PATH, MAX_CHARS)
    bits_tx = text_to_bits(input_text)
    symbols_tx = pam4_mod(bits_tx)

    print_simulation_parameters(
        input_text,
        bits_tx,
        symbols_tx,
        DEFAULT_DISTANCE,
        DEFAULT_NOISE_POWER,
    )

    main_results = simulate_all_links(
        symbols_tx,
        bits_tx,
        DEFAULT_DISTANCE,
        DEFAULT_NOISE_POWER,
    )
    print_main_results(main_results)

    noise_results = [
        (
            noise_power,
            simulate_all_links(symbols_tx, bits_tx, DEFAULT_DISTANCE, noise_power),
        )
        for noise_power in NOISE_LEVELS
    ]
    print_noise_variation(noise_results)

    plot_path_loss()


if __name__ == "__main__":
    main()

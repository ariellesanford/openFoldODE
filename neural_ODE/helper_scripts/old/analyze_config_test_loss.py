import re
import os
from typing import Dict, List, Tuple


def parse_memory_report(report_path: str) -> List[Dict]:
    """
    Parse a memory optimization report to extract loss values and configurations
    """
    configurations = []

    with open(report_path, 'r') as f:
        content = f.read()

    # Find the detailed configuration section
    detailed_section = content.split("Detailed Configuration Results:")[
        1] if "Detailed Configuration Results:" in content else ""

    # Parse each configuration
    config_blocks = detailed_section.split("Configuration ")[1:]  # Skip the first empty split

    for block in config_blocks:
        try:
            lines = block.strip().split('\n')

            # Extract configuration name
            name_line = lines[0] if lines else ""
            name = name_line.split(': ')[1] if ': ' in name_line else "Unknown"

            # Extract status
            status = "UNKNOWN"
            loss = None
            memory = None
            time = None

            for line in lines:
                if "Status:" in line:
                    status = line.split("Status: ")[1].strip()
                elif "Loss:" in line and "Results:" not in line:
                    try:
                        loss = float(line.split("Loss: ")[1].strip())
                    except:
                        pass
                elif "Max Memory:" in line:
                    try:
                        memory_str = line.split("Max Memory: ")[1].split(" MB")[0].strip()
                        memory = float(memory_str)
                    except:
                        pass
                elif "Time:" in line:
                    try:
                        time_str = line.split("Time: ")[1].split(" ms")[0].strip()
                        time = float(time_str)
                    except:
                        pass

            if status == "SUCCESS" and loss is not None:
                configurations.append({
                    'name': name,
                    'status': status,
                    'loss': loss,
                    'memory': memory,
                    'time': time
                })

        except Exception as e:
            print(f"Error parsing configuration block: {e}")
            continue

    return configurations


def analyze_config_results_with_baselines(config_results: List[Dict], baselines: Dict = None):
    """
    Analyze configuration test results and interpret them using data baselines
    """
    print("=== Memory Configuration Results Analysis ===\n")

    if not config_results:
        print("No successful configurations found!")
        return

    # Sort by loss (best first)
    sorted_configs = sorted(config_results, key=lambda x: x['loss'])

    print("Configuration Performance Ranking:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Name':<25} {'Loss':<10} {'Memory (MB)':<12} {'Time (ms)':<10} {'Assessment'}")
    print("-" * 80)

    for i, config in enumerate(sorted_configs, 1):
        memory_str = f"{config['memory']:.1f}" if config['memory'] else "N/A"
        time_str = f"{config['time']:.1f}" if config['time'] else "N/A"

        # Assess performance based on baselines if provided
        assessment = assess_loss_performance(config['loss'], baselines)

        print(f"{i:<5} {config['name'][:24]:<25} {config['loss']:<10.2f} {memory_str:<12} {time_str:<10} {assessment}")

    print("-" * 80)

    # Performance analysis
    best_config = sorted_configs[0]
    worst_config = sorted_configs[-1]

    print(f"\n=== Performance Analysis ===")
    print(f"Best configuration: {best_config['name']}")
    print(f"  Loss: {best_config['loss']:.2f}")
    print(f"  Memory: {best_config['memory']:.1f} MB" if best_config['memory'] else "  Memory: N/A")
    print(f"  Time: {best_config['time']:.1f} ms" if best_config['time'] else "  Time: N/A")

    print(f"\nWorst configuration: {worst_config['name']}")
    print(f"  Loss: {worst_config['loss']:.2f}")
    print(f"  Performance degradation: {worst_config['loss'] / best_config['loss']:.1f}x worse")

    # Memory vs Performance trade-offs
    if any(c['memory'] for c in sorted_configs):
        print(f"\n=== Memory vs Performance Trade-offs ===")

        # Find configurations with good memory efficiency
        memory_sorted = sorted([c for c in sorted_configs if c['memory']], key=lambda x: x['memory'])

        if memory_sorted:
            memory_efficient = memory_sorted[0]
            print(f"Most memory efficient: {memory_efficient['name']}")
            print(f"  Memory: {memory_efficient['memory']:.1f} MB")
            print(f"  Loss: {memory_efficient['loss']:.2f}")
            print(
                f"  Performance cost: {memory_efficient['loss'] / best_config['loss']:.1f}x worse loss for {best_config['memory'] / memory_efficient['memory']:.1f}x less memory")

    # Speed vs Performance trade-offs
    if any(c['time'] for c in sorted_configs):
        print(f"\n=== Speed vs Performance Trade-offs ===")

        time_sorted = sorted([c for c in sorted_configs if c['time']], key=lambda x: x['time'])

        if time_sorted:
            fastest = time_sorted[0]
            print(f"Fastest configuration: {fastest['name']}")
            print(f"  Time: {fastest['time']:.1f} ms")
            print(f"  Loss: {fastest['loss']:.2f}")
            print(
                f"  Performance cost: {fastest['loss'] / best_config['loss']:.1f}x worse loss for {best_config['time'] / fastest['time']:.1f}x faster")


def assess_loss_performance(loss_value: float, baselines: Dict = None) -> str:
    """
    Assess loss performance based on baselines
    """
    if baselines is None:
        # Without baselines, we can only give general guidance
        if loss_value < 20:
            return "Very Good"
        elif loss_value < 40:
            return "Good"
        elif loss_value < 60:
            return "Decent"
        else:
            return "Poor"

    # With baselines, give more precise assessment
    natural_var = baselines.get('natural_variation', float('inf'))
    mean_baseline = baselines.get('mean_baseline', float('inf'))
    zero_baseline = baselines.get('zero_baseline', float('inf'))

    if loss_value < natural_var:
        return "Excellent"
    elif loss_value < mean_baseline / 10:
        return "Very Good"
    elif loss_value < mean_baseline / 2:
        return "Good"
    elif loss_value < mean_baseline:
        return "Decent"
    elif loss_value < zero_baseline:
        return "Poor"
    else:
        return "Terrible"


def compare_multiple_reports(report_dir: str):
    """
    Compare results across multiple memory optimization reports
    """
    print("=== Comparing Multiple Configuration Reports ===\n")

    report_files = [f for f in os.listdir(report_dir) if
                    f.startswith('memory_optimization_report') and f.endswith('.txt')]

    if not report_files:
        print("No memory optimization reports found!")
        return

    all_results = {}

    for report_file in sorted(report_files):
        report_path = os.path.join(report_dir, report_file)
        print(f"Analyzing {report_file}...")

        configs = parse_memory_report(report_path)
        if configs:
            best_config = min(configs, key=lambda x: x['loss'])
            all_results[report_file] = {
                'best_loss': best_config['loss'],
                'best_config': best_config['name'],
                'num_successful': len(configs)
            }

    print(f"\nComparison across reports:")
    print("-" * 60)
    print(f"{'Report':<30} {'Best Loss':<12} {'Best Config':<25}")
    print("-" * 60)

    for report, results in sorted(all_results.items(), key=lambda x: x[1]['best_loss']):
        print(f"{report[:29]:<30} {results['best_loss']:<12.2f} {results['best_config'][:24]}")

    print("-" * 60)


def main():
    """
    Main function to analyze your configuration test results
    """
    # Path to your config test outputs
    config_dir = "/home/visitor/PycharmProjects/openFold/neural_ODE/config_test_outputs"

    # Find the most recent report
    report_files = [f for f in os.listdir(config_dir) if
                    f.startswith('memory_optimization_report') and f.endswith('.txt')]

    if not report_files:
        print("No memory optimization reports found!")
        print(f"Looking in: {config_dir}")
        return

    # Use the most recent report (or specify a particular one)
    latest_report = sorted(report_files)[-1]
    report_path = os.path.join(config_dir, latest_report)

    print(f"Analyzing: {latest_report}")
    print(f"Path: {report_path}\n")

    # Parse the configuration results
    config_results = parse_memory_report(report_path)

    if not config_results:
        print("No successful configurations found in the report!")
        return

    # If you've run the data baseline analysis, you can provide those baselines here
    # For now, we'll analyze without baselines
    baselines = None

    # TODO: Uncomment and fill in these values after running the data baseline analysis
    # baselines = {
    #     'natural_variation': 2.1,    # From your data baseline analysis
    #     'mean_baseline': 8.4,        # From your data baseline analysis
    #     'zero_baseline': 15.2        # From your data baseline analysis
    # }

    # Analyze the results
    analyze_config_results_with_baselines(config_results, baselines)

    # Compare with other reports if available
    if len(report_files) > 1:
        print(f"\n" + "=" * 60)
        compare_multiple_reports(config_dir)


if __name__ == "__main__":
    main()
# python3 circos_generator.py --csv path/to/your/input.csv --out_dir path/to/output_directory

import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from pycirclize import Circos
import argparse

# Define the genome lengths for the respective viruses (in base pairs)
genome_lengths = {
    'sars_cov_2': 29903,      # SARS-CoV-2 genome length
    'influenza_a': 13362,     # Influenza A genome length
    'rsv': 15200,             # Respiratory syncytial virus genome length
    'rhinovirus': 7200        # Rhinovirus genome length
}

def calculate_label_statistics(predictions, label_type):
    label_counts = defaultdict(int)
    label_confidence_sums = defaultdict(float)
    read_lengths = defaultdict(list)  # Track read lengths for each virus class

    for prediction in predictions:
        if label_type == 'virus':
            virus_label = prediction['Virus_Label']
            virus_conf = prediction['Virus_Confidence']
            read_length = len(prediction['Sequence'])  # Get read length
            if virus_label not in ('Other', 'unspecified'):  # Exclude 'Other' and 'unspecified' from the count
                label_counts[virus_label] += 1
                label_confidence_sums[virus_label] += virus_conf
                read_lengths[virus_label].append(read_length)
        elif label_type == 'variant':
            variant_label = prediction['Variant_Label']
            variant_conf = prediction['Variant_Confidence']
            if variant_label not in ('unknown_variant', 'nonRBD', 'unknown_LPAI', 'unspecified'):  # Exclude non-relevant labels
                label_counts[variant_label] += 1
                label_confidence_sums[variant_label] += variant_conf

    df = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
    df['Confidence_Sum'] = df['Label'].map(label_confidence_sums)
    df['Avg_Confidence'] = df['Confidence_Sum'] / df['Count']
    
    # Calculate the average read length for each virus
    avg_read_lengths = {label: np.mean(read_lengths[label]) if label in read_lengths else 0 for label in df['Label']}
    df['Avg_Read_Length'] = df['Label'].map(avg_read_lengths)

    # Calculate virus copy number if genome length is available
    df['Virus_Copy_Number'] = df.apply(lambda row: 
                                       (row['Count'] * row['Avg_Read_Length']) / genome_lengths.get(row['Label'], np.nan),
                                       axis=1)

    label_statistics = df.set_index('Label').to_dict(orient='index')
    
    return label_statistics

def analyze_predictions(output_csv_file, prediction_output_dir):
    df = pd.read_csv(output_csv_file)

    # Filter out unspecified virus labels before processing further
    df = df[df['Virus_Label'] != 'unspecified']

    virus_label_statistics = calculate_label_statistics(df.to_dict('records'), 'virus')
    sorted_virus_label_statistics = sorted(virus_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)

    total_classifications = sum(item[1]['Count'] for item in sorted_virus_label_statistics)
    
    # Calculate the percentages of the top two virus labels
    if len(sorted_virus_label_statistics) > 1:
        top_two_percentages = [
            sorted_virus_label_statistics[0][1]['Count'] / total_classifications * 100,
            sorted_virus_label_statistics[1][1]['Count'] / total_classifications * 100
        ]
    else:
        top_two_percentages = [sorted_virus_label_statistics[0][1]['Count'] / total_classifications * 100]

    # Display the top two virus labels if both have more than 5% of classifications
    if len(top_two_percentages) > 1 and all(pct > 5 for pct in top_two_percentages):
        for label, stats in sorted_virus_label_statistics[:2]:
            print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']*100:.4f}, "
                  f"Virus Copy Number: {stats['Virus_Copy_Number']:.4f}")
    else:
        # Display only the top label
        label, stats = sorted_virus_label_statistics[0]
        print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']*100:.4f}, "
              f"Virus Copy Number: {stats['Virus_Copy_Number']:.4f}")

    # Create a pie chart of the viral label distribution
    label_counts = df['Virus_Label'].value_counts().to_dict()
    labels = list(label_counts.keys())
    sizes = list(label_counts.values())
    
    # Calculate percentages
    sars_cov_2_percentage = label_counts.get('sars_cov_2', 0) / total_classifications * 100
    iav_percentage = label_counts.get('influenza_a', 0) / total_classifications * 100
    
    # Display SARS-CoV-2 VOC stats if classified as > 5% or if 100% classified as sars_cov_2
    if sars_cov_2_percentage > 5 or sars_cov_2_percentage == 100:
        cov2_predictions = df[df['Virus_Label'] == 'sars_cov_2']
        cov2_label_statistics = calculate_label_statistics(cov2_predictions.to_dict('records'), 'variant')
        sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)

        print("\nTop Variant Labels for sars_cov_2:")
        for label, stats in sorted_cov2_label_statistics[:1]:
            print(f"Variant Label: {label}, Average Confidence: {stats['Avg_Confidence']*100:.4f}")
    
    # Display IAV strain stats if classified as > 5% or if 100% classified as influenza_a
    if iav_percentage > 5 or iav_percentage == 100:
        iav_predictions = df[df['Virus_Label'] == 'influenza_a']
        iav_label_statistics = calculate_label_statistics(iav_predictions.to_dict('records'), 'variant')
        sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)

        print("\nTop Variant Labels for influenza_a:")
        for label, stats in sorted_iav_label_statistics[:1]:
            print(f"Variant Label: {label}, Average Confidence: {stats['Avg_Confidence']*100:.4f}")

    # Create a Circos plot of the viral label distribution
    label_counts = df['Virus_Label'].value_counts().to_dict()
    
    # Define sectors with names and sizes for virus labels
    sectors = {label: {'size': count, 'color': plt.cm.tab20(i / len(label_counts))} for i, (label, count) in enumerate(label_counts.items())}
    total_size = sum(v['size'] for v in sectors.values())

    # Initialize Circos with sectors and add space between them
    circos = Circos(
        {k: v['size'] for k, v in sectors.items()},
        space=8  # Adding space between sectors
    )
    circos.text(f"Virus_Labels", r=105, size=15, weight="bold", adjust_rotation=True, ha="center", va="center", orientation="horizontal")
    circos.text(f"Variant_Labels", r=50, size=15, weight="bold", adjust_rotation=True, ha="center", va="center", orientation="horizontal")
    
    # Add tracks to each sector with unique transparent colors and add labels inside
    for sector in circos.sectors:
        sector_name = sector.name
        sector_size = sectors[sector_name]['size']
        sector_color = sectors[sector_name]['color']
        percentage = (sector_size / total_size) * 100
        
        track = sector.add_track((60, 100), r_pad_ratio=20)
        track.axis(fc=sector_color, ec="black", alpha=0.5)
        
        # Add label with sector name and percentage inside the sector
        label = f"{sector_name}\n({percentage:.1f}%)"
        sector.text(label, r=65, size=15, color="black", ha="center", va="center", adjust_rotation=True, orientation="vertical")
        
        # Check for specific virus labels and add top variant sector
        if sector_name == 'sars_cov_2':
            cov2_predictions = df[df['Virus_Label'] == 'sars_cov_2']
            cov2_label_statistics = calculate_label_statistics(cov2_predictions.to_dict('records'), 'variant')
            sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)
            
            # Display only the top variant
            if sorted_cov2_label_statistics:
                top_variant, top_stats = sorted_cov2_label_statistics[0]
                var_percentage = (top_stats['Count'] / total_classifications) * 100
                var_color = plt.cm.Paired(0)  # Assign a distinct color for the top variant
                var_label = f"{top_variant}"
                
                variant_track = sector.add_track((0, 50), r_pad_ratio=20)
                variant_track.axis(fc=var_color, ec="black", alpha=0.5)
                variant_track.text(var_label, r=30, size=12, color="black", ha="center", va="center", adjust_rotation=True, orientation="vertical")

        elif sector_name == 'influenza_a':
            iav_predictions = df[df['Virus_Label'] == 'influenza_a']
            iav_label_statistics = calculate_label_statistics(iav_predictions.to_dict('records'), 'variant')
            sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)
            
            # Display only the top variant
            if sorted_iav_label_statistics:
                top_variant, top_stats = sorted_iav_label_statistics[0]
                var_percentage = (top_stats['Count'] / total_classifications) * 100
                var_color = plt.cm.Paired(0.5)  # Assign a distinct color for the top variant
                var_label = f"{top_variant}"
                
                variant_track = sector.add_track((0, 50), r_pad_ratio=20)
                variant_track.axis(fc=var_color, ec="black", alpha=0.5)
                variant_track.text(var_label, r=30, size=12, color="black", ha="center", va="center", adjust_rotation=True, orientation="vertical")

    # Plot the Circos plot
    circos.plotfig()
    plt.show()
    
    # Extract the base name from the input CSV file to save the JPEG file
    base_filename = os.path.basename(output_csv_file).rsplit('.', 1)[0]
    
    # Save the Circos plot as a JPEG file with the extracted base name
    plot_filename = os.path.join(prediction_output_dir, f"{base_filename}_circos_plot.jpeg")
    circos.savefig(plot_filename, dpi=300) 
    
if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze viral predictions and create visualizations.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the input CSV file containing prediction data.")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save the analysis output and plots.")

    # Parse the arguments
    args = parser.parse_args()

    # Run the analysis with the provided arguments
    analyze_predictions(args.csv, args.out_dir)
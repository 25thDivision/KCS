import csv

noises = ["realistic/dp0.001_mf0.01_rf0.01_gd0.008", "realistic/dp0.005_mf0.02_rf0.02_gd0.015", "realistic/dp0.01_mf0.05_rf0.05_gd0.01"]
models = ["APPNP", "CNN", "GAT", "GCN", "GCNII", "GNN", "GraphTransformer", "GraphMamba"]
columns = ["Model", "Distance", "Error_Rate(p)", "Error_Type", "Noise", "Best_ECR(%)", "Accuracy(%)",
               "Inference_Time(ms)", "Epochs", "Learning_Rate", "Train_Loss", "Val_Loss"]


with open("stim_total.csv", "w", newline='') as output:
    writer = csv.writer(output)
    writer.writerow(columns)
    for noise in noises:
        for model in models:
            with open(f"color_code/{noise}/benchmark_{model}.csv") as input:
                for line in input:
                    if line.startswith("Distance"):
                        continue
                    newline = line.strip().split(",")
                    writer.writerow([model, newline[0], newline[1], newline[2], noise, newline[3], newline[4], newline[5], newline[6], newline[7], newline[8], newline[9]])  # Adjust column order as needed

            print(f"Finished processing {noise} for {model}")
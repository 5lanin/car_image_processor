from datasets import load_dataset

# Download/Loading of the dataset
dataset_name = "DamianBoborzi/car_images"
print("\n START DOWNLOAD \n")
try:
    dataset = load_dataset(dataset_name)

    # Zeige die Struktur des Datensatzes an
    print("Datensatz erfolgreich geladen! \n")
except Exception as e:
    print(f"Fehler beim Laden des Datensatzes: {e}")


# Examining the data set
print(dataset, "\n")
print(dataset["train"][0], "\n")
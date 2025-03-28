import subprocess

def executar_pipeline():
    print("\n=== Etapa 1: Preparação dos Dados ===")
    subprocess.run(["python", "../DataPrep/data_preparation.py"], check=True)

    print("\n=== Etapa 2: Treinamento do Modelo ===")
    subprocess.run(["python", "../Model/train_model.py"], check=True)

    print("\n=== Etapa 3: Aplicação em Produção ===")
    subprocess.run(["python", "aplicacao.py"], check=True)

    print("\n✅ Pipeline executado com sucesso!")

if __name__ == "__main__":
    executar_pipeline()

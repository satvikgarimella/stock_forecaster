from data.data_fetcher import fetch_data
from utils.data_preprocessing import preprocess_data, create_sequences
from models.train import train_model
from utils.visualization import plot_predictions

def main():
    # Step 1: Fetch Data
    data = fetch_data('AAPL', '2015-01-01', '2023-01-01')

    # Step 2: Preprocess Data
    scaled_data, scaler = preprocess_data(data)
    X_train, y_train, X_test, y_test = create_sequences(scaled_data)

    # Step 3: Train Model
    model = train_model(X_train, y_train)

    # Step 4: Make Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Step 5: Plot Results
    plot_predictions(y_test, predictions, scaler)

if __name__ == "__main__":
    main()
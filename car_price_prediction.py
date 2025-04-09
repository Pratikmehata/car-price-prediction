{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "649b742a-da03-4a6b-bf05-a8aa2a74cc23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[31mERROR: Could not find a version that satisfies the requirement pickle (from versions: none)\u001b[0m\u001b[31m\n",
      "\u001b[0m\u001b[31mERROR: No matching distribution found for pickle\u001b[0m\u001b[31m\n",
      "\u001b[0mNote: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pickle\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "24a1406d-cc8b-4f5f-90c9-d9ad29b3d163",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Car_Name  Year  Selling_Price  Present_Price  Kms_Driven Fuel_Type  \\\n",
      "0     ritz  2014           3.35           5.59       27000    Petrol   \n",
      "1      sx4  2013           4.75           9.54       43000    Diesel   \n",
      "2     ciaz  2017           7.25           9.85        6900    Petrol   \n",
      "3  wagon r  2011           2.85           4.15        5200    Petrol   \n",
      "4    swift  2014           4.60           6.87       42450    Diesel   \n",
      "\n",
      "  Seller_Type Transmission  Owner  \n",
      "0      Dealer       Manual      0  \n",
      "1      Dealer       Manual      0  \n",
      "2      Dealer       Manual      0  \n",
      "3      Dealer       Manual      0  \n",
      "4      Dealer       Manual      0  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('/home/pratik/Downloads/car data.csv')  # Replace with your dataset path\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d1105195-92a9-449e-bc8a-1549753028cd",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'fueltype'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mKeyError\u001b[39m                                  Traceback (most recent call last)",
      "\u001b[36mFile \u001b[39m\u001b[32m~/myenv/lib/python3.12/site-packages/pandas/core/indexes/base.py:3805\u001b[39m, in \u001b[36mIndex.get_loc\u001b[39m\u001b[34m(self, key)\u001b[39m\n\u001b[32m   3804\u001b[39m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[32m-> \u001b[39m\u001b[32m3805\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43m_engine\u001b[49m\u001b[43m.\u001b[49m\u001b[43mget_loc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mcasted_key\u001b[49m\u001b[43m)\u001b[49m\n\u001b[32m   3806\u001b[39m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n",
      "\u001b[36mFile \u001b[39m\u001b[32mindex.pyx:167\u001b[39m, in \u001b[36mpandas._libs.index.IndexEngine.get_loc\u001b[39m\u001b[34m()\u001b[39m\n",
      "\u001b[36mFile \u001b[39m\u001b[32mindex.pyx:196\u001b[39m, in \u001b[36mpandas._libs.index.IndexEngine.get_loc\u001b[39m\u001b[34m()\u001b[39m\n",
      "\u001b[36mFile \u001b[39m\u001b[32mpandas/_libs/hashtable_class_helper.pxi:7081\u001b[39m, in \u001b[36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[39m\u001b[34m()\u001b[39m\n",
      "\u001b[36mFile \u001b[39m\u001b[32mpandas/_libs/hashtable_class_helper.pxi:7089\u001b[39m, in \u001b[36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[39m\u001b[34m()\u001b[39m\n",
      "\u001b[31mKeyError\u001b[39m: 'fueltype'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[31mKeyError\u001b[39m                                  Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[6]\u001b[39m\u001b[32m, line 7\u001b[39m\n\u001b[32m      5\u001b[39m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01msklearn\u001b[39;00m\u001b[34;01m.\u001b[39;00m\u001b[34;01mpreprocessing\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m LabelEncoder\n\u001b[32m      6\u001b[39m le = LabelEncoder()\n\u001b[32m----> \u001b[39m\u001b[32m7\u001b[39m df[\u001b[33m'\u001b[39m\u001b[33mfueltype\u001b[39m\u001b[33m'\u001b[39m] = le.fit_transform(\u001b[43mdf\u001b[49m\u001b[43m[\u001b[49m\u001b[33;43m'\u001b[39;49m\u001b[33;43mfueltype\u001b[39;49m\u001b[33;43m'\u001b[39;49m\u001b[43m]\u001b[49m)\n\u001b[32m      8\u001b[39m df[\u001b[33m'\u001b[39m\u001b[33maspiration\u001b[39m\u001b[33m'\u001b[39m] = le.fit_transform(df[\u001b[33m'\u001b[39m\u001b[33maspiration\u001b[39m\u001b[33m'\u001b[39m])\n\u001b[32m      9\u001b[39m \u001b[38;5;66;03m# Repeat for other categorical columns\u001b[39;00m\n\u001b[32m     10\u001b[39m \n\u001b[32m     11\u001b[39m \u001b[38;5;66;03m# Feature selection\u001b[39;00m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/myenv/lib/python3.12/site-packages/pandas/core/frame.py:4102\u001b[39m, in \u001b[36mDataFrame.__getitem__\u001b[39m\u001b[34m(self, key)\u001b[39m\n\u001b[32m   4100\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m.columns.nlevels > \u001b[32m1\u001b[39m:\n\u001b[32m   4101\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m._getitem_multilevel(key)\n\u001b[32m-> \u001b[39m\u001b[32m4102\u001b[39m indexer = \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43mcolumns\u001b[49m\u001b[43m.\u001b[49m\u001b[43mget_loc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\n\u001b[32m   4103\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[32m   4104\u001b[39m     indexer = [indexer]\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/myenv/lib/python3.12/site-packages/pandas/core/indexes/base.py:3812\u001b[39m, in \u001b[36mIndex.get_loc\u001b[39m\u001b[34m(self, key)\u001b[39m\n\u001b[32m   3807\u001b[39m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(casted_key, \u001b[38;5;28mslice\u001b[39m) \u001b[38;5;129;01mor\u001b[39;00m (\n\u001b[32m   3808\u001b[39m         \u001b[38;5;28misinstance\u001b[39m(casted_key, abc.Iterable)\n\u001b[32m   3809\u001b[39m         \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28many\u001b[39m(\u001b[38;5;28misinstance\u001b[39m(x, \u001b[38;5;28mslice\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m x \u001b[38;5;129;01min\u001b[39;00m casted_key)\n\u001b[32m   3810\u001b[39m     ):\n\u001b[32m   3811\u001b[39m         \u001b[38;5;28;01mraise\u001b[39;00m InvalidIndexError(key)\n\u001b[32m-> \u001b[39m\u001b[32m3812\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01merr\u001b[39;00m\n\u001b[32m   3813\u001b[39m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:\n\u001b[32m   3814\u001b[39m     \u001b[38;5;66;03m# If we have a listlike key, _check_indexing_error will raise\u001b[39;00m\n\u001b[32m   3815\u001b[39m     \u001b[38;5;66;03m#  InvalidIndexError. Otherwise we fall through and re-raise\u001b[39;00m\n\u001b[32m   3816\u001b[39m     \u001b[38;5;66;03m#  the TypeError.\u001b[39;00m\n\u001b[32m   3817\u001b[39m     \u001b[38;5;28mself\u001b[39m._check_indexing_error(key)\n",
      "\u001b[31mKeyError\u001b[39m: 'fueltype'"
     ]
    }
   ],
   "source": [
    "# Handle missing values\n",
    "df.dropna(inplace=True)\n",
    "\n",
    "# Convert categorical data\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "df['Fuel_type'] = le.fit_transform(df['Fuel_type'])\n",
    "df['aspiration'] = le.fit_transform(df['aspiration'])\n",
    "# Repeat for other categorical columns\n",
    "\n",
    "# Feature selection\n",
    "X = df[['fueltype', 'aspiration', 'wheelbase', 'curbweight', \n",
    "        'enginesize', 'horsepower', 'price']]\n",
    "y = df['price']\n",
    "\n",
    "# Split data\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d4fdd842-c5db-442b-8728-29d1beceb2cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[31mERROR: Could not find a version that satisfies the requirement pickle (from versions: none)\u001b[0m\u001b[31m\n",
      "\u001b[0m\u001b[31mERROR: No matching distribution found for pickle\u001b[0m\u001b[31m\n",
      "\u001b[0mNote: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pickle\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "78eafaa0-c1b4-4e7b-8971-0a6a321f8ec0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: ₹0.61 Lakhs\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from xgboost import XGBRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "import pickle\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('/home/pratik/Downloads/car data.csv')\n",
    "\n",
    "# Data Preprocessing\n",
    "# Convert categorical data\n",
    "le = LabelEncoder()\n",
    "df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])\n",
    "df['Seller_Type'] = le.fit_transform(df['Seller_Type'])\n",
    "df['Transmission'] = le.fit_transform(df['Transmission'])\n",
    "\n",
    "# Feature selection\n",
    "X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', \n",
    "        'Seller_Type', 'Transmission', 'Owner']]\n",
    "y = df['Selling_Price']\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Initialize and train model\n",
    "model = XGBRegressor()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate\n",
    "predictions = model.predict(X_test)\n",
    "mae = mean_absolute_error(y_test, predictions)\n",
    "print(f'Mean Absolute Error: ₹{mae:.2f} Lakhs')\n",
    "\n",
    "# Save model\n",
    "with open('car_price_model.pkl', 'wb') as file:\n",
    "    pickle.dump(model, file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3e02bc12-e7f7-4841-b3e0-d480acd0f0e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type',\n",
    "        'Seller_Type', 'Transmission', 'Owner']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "7f76c310-1b77-47d9-9ace-0b235979ae06",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

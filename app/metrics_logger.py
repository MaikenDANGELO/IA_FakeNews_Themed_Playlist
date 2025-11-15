# metrics_logger.py

import json
from datetime import datetime
import numpy as np

def save_metrics(report_filename, class_report_dict, conf_matrix):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metrics_data = {
        "timestamp": timestamp,
        "classification_report": class_report_dict,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    try:
        with open(report_filename, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"\nMétriques enregistrées avec succès dans: {report_filename}")
    except Exception as e:
        print(f"\nErreur lors de l’enregistrement des mesures: {e}")
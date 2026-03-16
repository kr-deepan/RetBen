def get_triage(class_id, confidence):
    """
    Implements clinical triage decision logic.
    
    Logic:
    Grade 0 or 1 -> "Monitor"
    Grade 2, 3, or 4 -> "Refer to specialist"
    
    Risk score is the confidence of the predicted class.
    
    Args:
        class_id (int): Predicted DR class (0-4)
        confidence (float): Confidence score of the prediction (0.0 to 1.0)
        
    Returns:
        dict: {"triage": str, "risk_score": float}
    """
    
    if class_id in [0, 1]:
        triage_recommendation = "Monitor"
    elif class_id in [2, 3, 4]:
        triage_recommendation = "Refer to specialist"
    else:
        triage_recommendation = "Unknown"
        
    # As requested, risk score = probability of predicted class
    risk_score = round(confidence, 4)
    
    return {
        "triage": triage_recommendation,
        "risk_score": risk_score
    }

if __name__ == '__main__':
    # Test cases
    print(get_triage(0, 0.95)) # Monitor
    print(get_triage(2, 0.82)) # Refer
    print(get_triage(4, 0.99)) # Refer

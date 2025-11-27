import tkinter as tk
from tkinter import ttk
import numpy as np # Used for simulated ML calculations (though not explicitly used for calculation here, it sets the context)

# =================================================================================
# --- AI/ML TOPICS: FEATURE DEFINITION AND IMPORTANCE (Expert-Defined Weights) ---
# =================================================================================
# Define the evaluation criteria (Features for the ML model). Each feature has
# predefined discrete options and corresponding point values, simulating expert
# knowledge. These points represent the *initial* feature weights in a rules-based system.
CRITERIA = [
    ("Sight", [
        # Feature 1: Clarity
        ("Clarity", [("Brilliant", 3), ("Hazy", 1), ("Faulty", 0)]),
        # Feature 2: Colora
        ("Color", [("Vibrant", 4), ("Slightly Off", 2), ("Oxidized", 0)])
    ]),
    ("Smell", [
        # Feature 3: Critical Faults - Given a high inherent weight (20 points max)
        # to ensure the rules-based system fails on any fault.
        ("Faults (Critical)", [("No Faults", 20), ("Faulty", 0)]),
        # Feature 4: Intensity
        ("Intensity", [("Powerful", 6), ("Medium", 3), ("Faint", 1)]),
        # Feature 5: Complexity
        ("Complexity", [("Complex", 10), ("Some Layers", 5), ("Simple", 2)])
    ]),
    ("Taste", [
        # Feature 6: Balance - The most heavily weighted feature (30 points max) in
        # the traditional system, simulating high feature importance based on domain expertise.
        ("Balance (Critical)", [("Perfectly Balanced", 30), ("Some Imbalance", 15), ("Dominant Component", 5)]),
        # Feature 7: Finish
        ("Finish", [("Long", 30), ("Medium", 15), ("Short", 5)]),
        # Feature 8: Flavor Intensity
        ("Flavor Intensity", [("Pronounced", 4), ("Medium", 2), ("Light", 1)])
    ]),
]

# Calculate maximum possible score for the Rules-Based System.
MAX_SCORE = sum(max(points for _, points in options) for _, questions in CRITERIA for _, options in questions)

# ===================================================================================
# --- AI/ML TOPICS: MODEL WEIGHTS (Simulated Learned Feature Importance) ---
# ===================================================================================
# Simulating 'Learned Weights' from a trained ML model (e.g., a Linear Regression or DNN layer).
# These weights are different from the simple point values and indicate which features
# the ML model found most predictive of high quality in historical data.
ML_FEATURE_WEIGHTS = {
    "Clarity": 0.5,
    "Color": 0.3,
    "Faults (Critical)": 4.0, # Very high weight for critical faults, ensuring a strong penalty.
    "Intensity": 0.8,
    "Complexity": 1.2,
    "Balance (Critical)": 5.5, # Highest weight: ML confirmed that 'Balance' is the most important predictor.
    "Finish": 2.0,
    "Flavor Intensity": 0.6
}

# Calculate the theoretical maximum score for the *ML model's weighted sum*.
# This is used for normalization (simulating a sigmoid/softmax activation).
ML_MAX_PREDICTED_SCORE = sum(max(points for _, points in options) * ML_FEATURE_WEIGHTS.get(label, 1.0)
                             for _, questions in CRITERIA for label, options in questions)

# The cutoff (Threshold) for classification, simulating a learned decision boundary.
# A probability above this threshold is classified as 'Good Wine'.
ML_GOOD_WINE_THRESHOLD = 0.75 

class MLWinePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍷 ML-Enhanced Wine Scorecard")
        self.vars = {}
        
        # UI Element holders for results
        self.rules_score_label = None
        self.rules_verdict_label = None
        self.ml_prob_label = None
        self.ml_verdict_label = None

        # --- NEW UI Styling & Setup ---
        # Define a consistent theme/color palette
        self.bg_color = "#f4f7f9" # Light Background
        self.primary_color = "#800020" # Deep Burgundy
        self.secondary_color = "#b02a37" # Red Accent
        self.critical_color = "#cc0000" # Error Red
        self.font_header = ("Poppins", 18, "bold")
        self.font_section = ("Poppins", 14, "bold")
        self.font_label = ("Arial", 10, "normal")

        root.configure(bg=self.bg_color)
        
        # Main Title
        tk.Label(root, text="🍷 Professional Quality Assessment (Dual System)", 
                 font=("Poppins", 22, "bold"), fg=self.primary_color, bg=self.bg_color, pady=10).pack(fill="x")
        
        # Input Container
        container = tk.Frame(root, bg=self.bg_color)
        container.pack(padx=15, pady=5, expand=True, fill="both")

        # Layout frames (split into left/right for better organization)
        left = tk.Frame(container, bg=self.bg_color)
        left.pack(side="left", fill="both", expand=True, padx=10)
        right = tk.Frame(container, bg=self.bg_color)
        right.pack(side="right", fill="both", expand=True, padx=10)
        
        # Populate input widgets based on CRITERIA (Features)
        for i, (section, questions) in enumerate(CRITERIA):
            frame = left if i < 2 else right # Place 'Taste' on the right
            self.make_section(frame, section, questions)

        self.result_panel(root)
        self.update_score()

    def make_section(self, parent, section, questions):
        """Creates the LabelFrame for each feature section (Sight, Smell, Taste)."""
        # Style the section title box
        box = tk.LabelFrame(parent, text=f"Section: {section.upper()}", 
                            font=self.font_section, fg=self.primary_color, 
                            bg="#ffffff", bd=3, relief=tk.RIDGE, padx=10, pady=5)
        box.pack(fill="x", padx=5, pady=12)
        
        for label, options in questions:
            frame = tk.Frame(box, bg="#ffffff")
            frame.pack(fill="x", pady=5, padx=5, side=tk.TOP)
            
            # Style for the specific feature label
            style_fg = self.critical_color if "Critical" in label else self.secondary_color
            
            # Use improved labels for features
            display_label = label.replace("(Critical)", "").strip()
            tk.Label(frame, text=f"• {display_label}:", 
                     font=("Arial", 10, "bold"), fg=style_fg, bg="#ffffff").pack(anchor="w", pady=(5, 2))
            
            btns = tk.Frame(frame, bg="#ffffff")
            btns.pack(anchor="w", pady=1)
            
            # Initialize the variable with the first option's score
            var = tk.IntVar(value=options[0][1])
            self.vars[label] = var # Store a reference to the variable, using the feature label as key
            
            # Use ttk.Style for better looking Radiobuttons
            style = ttk.Style()
            style.configure("T.TRadiobutton", font=self.font_label, background="#ffffff", foreground=self.secondary_color)
            style.map("T.TRadiobutton", 
                      background=[('active', '#f0f0f0'), ('selected', self.primary_color)],
                      foreground=[('selected', 'white')])

            # Create a Radiobutton for each option
            for txt, val in options:
                # Use ttk.Radiobutton for modern appearance
                ttk.Radiobutton(btns, text=f"{txt} ({val} pts)", variable=var, value=val,
                                command=self.update_score, style="T.TRadiobutton").pack(side="left", padx=8, ipadx=2)

    def result_panel(self, parent):
        """Sets up the UI elements for displaying the results from both systems."""
        panel = tk.LabelFrame(parent, text="Final Assessment Results", 
                              font=self.font_section, fg=self.primary_color, bg="#ffffff", bd=3)
        panel.pack(padx=15, pady=15, fill="x")
        
        # --- Display Panel for Rules-Based Score ---
        rules_frame = tk.LabelFrame(panel, text="1. Expert Score (Rules-Based)", 
                                    font=("Arial", 12, "bold"), fg="#0056b3", bg="#eef2f5", bd=1)
        rules_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        tk.Label(rules_frame, text="Total Points (Out of 107):", font=("Arial", 11), bg="#eef2f5").pack(pady=5)
        self.rules_score_label = tk.Label(rules_frame, text="", font=("Poppins", 16, "bold"), fg="#0056b3", bg="#eef2f5")
        self.rules_score_label.pack()
        
        tk.Label(rules_frame, text="Verdict:", font=("Arial", 11), bg="#eef2f5").pack(pady=(10, 2))
        self.rules_verdict_label = tk.Label(rules_frame, text="", font=("Arial", 12, "bold"), fg="#333333", bg="#eef2f5", wraplength=250)
        self.rules_verdict_label.pack(pady=5)
        
        # --- Display Panel for ML Model Prediction ---
        ml_frame = tk.LabelFrame(panel, text="2. ML Prediction (Probabilistic)", 
                                 font=("Arial", 12, "bold"), fg=self.secondary_color, bg="#eef2f5", bd=1)
        ml_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # AI/ML Topic: Softmax/Sigmoid Output (Probability)
        tk.Label(ml_frame, text="Predicted Probability (Simulated):", font=("Arial", 11), bg="#eef2f5").pack(pady=5)
        self.ml_prob_label = tk.Label(ml_frame, text="", font=("Poppins", 16, "bold"), fg=self.secondary_color, bg="#eef2f5")
        self.ml_prob_label.pack()
        
        # AI/ML Topic: Classification (Decision Boundary)
        tk.Label(ml_frame, text=f"(Quality Threshold: >{int(ML_GOOD_WINE_THRESHOLD*100)}%)", 
                 font=("Arial", 9, "italic"), fg="#555555", bg="#eef2f5").pack()
        
        tk.Label(ml_frame, text="Classification:", font=("Arial", 11), bg="#eef2f5").pack(pady=(10, 2))
        self.ml_verdict_label = tk.Label(ml_frame, text="", font=("Arial", 12, "bold"), fg="#333333", bg="#eef2f5", wraplength=250)
        self.ml_verdict_label.pack(pady=5)

    # ===================================================================================
    # --- AI/ML TOPICS: MODEL INFERENCE (SIMULATION) ---
    # (LOGIC UNCHANGED)
    # ===================================================================================
    def predict_with_ml_model(self, scores):
        """
        Simulates the forward pass (inference) of a simple Machine Learning model.
        This demonstrates the concept of a model using learned feature weights
        to calculate a weighted sum, which is then passed through an activation
        function (simulated by normalization) for binary classification.
        """
        weighted_sum = 0
        for label, score in scores.items():
            # Apply the simulated learned weight for each feature, simulating 'W * X'
            weight = ML_FEATURE_WEIGHTS.get(label, 1.0)
            weighted_sum += score * weight

        # Normalize the weighted sum to a 0-1 range based on the theoretical max.
        # This simulates the final 'activation' layer (like a Sigmoid function in a DNN
        # used for binary classification). Probability = Sigmoid(Weighted_Sum) ~ Weighted_Sum / MAX_Weighted_Sum.
        probability = weighted_sum / ML_MAX_PREDICTED_SCORE
        
        # Clamp the probability to be strictly between 0 and 1.
        probability = max(0.0, min(1.0, probability))
        
        # --- Decision Boundary/Classification Logic ---
        # The ML model must also incorporate critical domain rules.
        if scores.get("Faults (Critical)") == 0:
            # Override: Critical fault leads to guaranteed failure, regardless of other high scores.
            verdict = "ML Prediction: UNACCEPTABLE (Faulty)"
            color = "#b91c1c"
            probability = 0.05 # Override to a very low probability (near zero)
        elif probability >= ML_GOOD_WINE_THRESHOLD:
            # Pass: The probability exceeds the learned decision boundary/threshold.
            verdict = "ML Prediction: HIGH QUALITY (Passes Threshold)"
            color = "#006400"
        else:
            # Fail: The probability is below the threshold.
            verdict = "ML Prediction: LOW/MEDIUM QUALITY (Fails Threshold)"
            color = "#e8781b"
            
        return probability, verdict, color

    def update_score(self):
        """
        Executed on any button click to recalculate and update both prediction systems.
        """
        # 1. Get the current feature scores (the input vector 'X')
        scores = {k: v.get() for k, v in self.vars.items()}
        total_rules_score = sum(scores.values())

        # --- RULES-BASED SYSTEM LOGIC (Deterministic) ---
        if scores.get("Faults (Critical)", 1) == 0:
            rules_verdict, rules_color = "🔴 Faulty / Unworthy", self.critical_color
            total_rules_score = 0
        elif total_rules_score >= 90:
            rules_verdict, rules_color = "🌟 Outstanding (Expert Consensus)", "#008e2f"
        elif total_rules_score >= 65:
            rules_verdict, rules_color = "✅ Very Good (Strong Q.A. Score)", "#3c721a"
        elif total_rules_score >= 40:
            rules_verdict, rules_color = "👍 Acceptable (Meets Minimum Standard)", "#b8a100"
        else:
            rules_verdict, rules_color = "🔻 Poor (Fails Standard)", "#e8781b"
            
        # Update Rules-Based UI
        self.rules_score_label.config(text=f"{total_rules_score}", fg=rules_color)
        self.rules_verdict_label.config(text=rules_verdict, fg=rules_color)
        
        # --- ML INFERENCE SYSTEM LOGIC ---
        # Call the simulated ML model's inference function with the feature scores.
        ml_probability, ml_verdict, ml_color = self.predict_with_ml_model(scores)
        
        # Update ML-Based UI
        self.ml_prob_label.config(text=f"{ml_probability:.2f}", fg=ml_color)
        self.ml_verdict_label.config(text=ml_verdict, fg=ml_color)


if __name__ == "__main__":
    root = tk.Tk()
    # Attempt to use Poppins font if available for a modern look
    try:
        root.option_add("*Font", "Poppins 10")
    except tk.TclError:
        pass # Fallback to default font if Poppins is not installed
    
    app = MLWinePredictorApp(root)
    # The main loop keeps the application running and listening for user events (like button clicks).
    root.mainloop()
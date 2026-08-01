from typing import Optional, Dict, Any

class DriveLMPromptBuilder:
    """
    Constructs Graph-of-Thought (GoT) prompt structures for DriveLM tasks,
    chaining spatial perception, object prediction, and vehicle planning.
    """
    
    SYSTEM_PROMPT = (
        "You are an autonomous vehicle driver AI operating in a multi-camera perception setup. "
        "Analyze the provided image frames and answer the spatial reasoning, object detection, "
        "and vehicle maneuver planning questions step-by-step."
    )

    def __init__(self):
        pass

    def build_graph_reasoning_prompt(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None,
        output_format: str = "JSON"
    ) -> str:
        """
        Builds a structured prompt following DriveLM DAG dependencies.
        """
        prompt = f"{self.SYSTEM_PROMPT}\n\n"
        
        if context:
            prompt += f"### Environmental Context:\n{context}\n\n"
            
        prompt += (
            "### Reasoning Task:\n"
            "Follow the Graph-of-Thought sequence:\n"
            "1. [Perception]: Identify key objects, traffic lights, and road agents in view.\n"
            "2. [Prediction]: Predict potential trajectories or state changes for key agents.\n"
            "3. [Planning]: Determine the optimal, safe maneuver for the ego vehicle.\n\n"
            f"Question: {question}\n\n"
        )
        
        if output_format.upper() == "JSON":
            prompt += (
                "Provide your output STRICTLY in valid JSON format with keys: "
                "`\"perception\"`, `\"prediction\"`, and `\"planning\"`."
            )
        else:
            prompt += "Provide a detailed step-by-step text explanation."

        return prompt
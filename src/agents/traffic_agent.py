# src/agents/traffic_agent.py
import datetime

class TrafficControlTools:
    """Simulated execution tools for municipal traffic control systems."""
    
    @staticmethod
    def set_ramp_metering(sensor_id: str, restriction_pct: int):
        return {
            "status": "SUCCESS",
            "action": "Ramp Metering Adjusted",
            "target": sensor_id,
            "details": f"Flow restricted by {restriction_pct}% to prevent main-line congestion spillback."
        }
        
    @staticmethod
    def update_vms_board(board_id: str, message: str):
        return {
            "status": "SUCCESS",
            "action": "Variable Message Sign Updated",
            "target": board_id,
            "details": f"Overhead sign now displaying: '{message}'"
        }
        
    @staticmethod
    def deploy_v2x_speed_cap(zone_id: str, target_speed_mph: int):
        return {
            "status": "SUCCESS",
            "action": "V2X Speed Harmonization Active",
            "target": zone_id,
            "details": f"Broadcasted {target_speed_mph} mph advisory cap to connected vehicle platoons."
        }

    @staticmethod
    def dispatch_incident_response(sensor_id: str, priority: str):
        return {
            "status": "SUCCESS",
            "action": "Incident Response Dispatched",
            "target": sensor_id,
            "details": f"Emergency Response Unit dispatched with {priority} priority."
        }

class AgentDecisionEngine:
    """ReAct (Reasoning + Acting) Agent Engine for Traffic Digital Twin."""
    
    def __init__(self):
        self.tools = TrafficControlTools()
        
    def evaluate_and_plan(self, zone: str, congestion_level: int, macro_demand: float, bottleneck_sensors: list):
        """Generates structured reasoning steps and recommended tool execution sequence."""
        thoughts = []
        proposed_actions = []
        
        # Step 1: Observation
        thoughts.append(f"🔍 **Observation:** Zone '{zone}' is experiencing a {congestion_level}% demand surge with regional demand at {int(macro_demand):,} vehicles/hr.")
        
        # Step 2: Diagnostic Reasoning & Counterfactual Planning
        if congestion_level < 35:
            thoughts.append("💡 **Reasoning:** Traffic flow is optimal. No aggressive intervention required.")
            proposed_actions.append(self.tools.update_vms_board(f"VMS-{zone[:3]}-01", "TRAFFIC FLOW NORMAL - DRIVE SAFELY"))
            
        elif congestion_level < 70:
            thoughts.append(f"💡 **Reasoning:** Impending bottleneck at sensors {bottleneck_sensors}. Main-line velocity dropping. Counterfactual test indicates 20% ramp reduction restores 12 mph average speed.")
            proposed_actions.append(self.tools.set_ramp_metering(bottleneck_sensors[0] if bottleneck_sensors else "Sensor #88", 20))
            proposed_actions.append(self.tools.update_vms_board(f"VMS-{zone[:3]}-01", "MODERATE DELAYS AHEAD - RAMP METERING ACTIVE"))
            
        else:
            thoughts.append(f"💡 **Reasoning:** Severe gridlock detected across {len(bottleneck_sensors)} critical nodes. Counterfactual model predicts severe spillback without immediate V2X speed caps and emergency ramp metering.")
            proposed_actions.append(self.tools.set_ramp_metering(bottleneck_sensors[0] if bottleneck_sensors else "Sensor #42", 40))
            proposed_actions.append(self.tools.deploy_v2x_speed_cap(zone, 30))
            proposed_actions.append(self.tools.update_vms_board(f"VMS-{zone[:3]}-01", "CRITICAL CONGESTION - REDUCE SPEED TO 30 MPH"))
            proposed_actions.append(self.tools.dispatch_incident_response(bottleneck_sensors[0] if bottleneck_sensors else "Sensor #42", "HIGH"))
            
        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "thoughts": thoughts,
            "actions": proposed_actions
        }
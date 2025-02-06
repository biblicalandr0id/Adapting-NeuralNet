# brainui.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
import random
import torch
import logging
import pyperclip  # For copy-paste functionality
from brain import Brain, BrainState

class BrainCore:
    def __init__(self):
        self.thought_patterns = torch.rand(20, 20)
        self.emotional_state = torch.zeros(5)  # [curiosity, satisfaction, uncertainty, focus, drive]
        self.current_focus = None
        self.thought_history = []
        self.learning_rate = 0.1
        
    def process_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Convert input to vector
        try:
            input_vector = torch.tensor(list(input_data.values())).float()
        except Exception as e:
            logging.error(f"Error processing input data, please ensure all input data values are numbers: {input_data}, Error: {e}")
            input_vector = torch.zeros(len(input_data))
        
        # Process through thought patterns
        processed = self.thought_patterns @ input_vector
        
        # Update emotional state
        self.emotional_state += torch.rand(5) * 0.1 - 0.05
        self.emotional_state = torch.clip(self.emotional_state, 0, 1)
        
        return {
            "processed_thought": processed.tolist(),
            "emotional_state": self.emotional_state.tolist(),
            "focus": self.current_focus
        }

class BrainInterface(tk.Tk):
    def __init__(self, mind, dna_guide):
        super().__init__()
        
        self.brain = BrainCore()
        self.mind = mind
        self.dna_guide = dna_guide
        self.mind_state = None # Store state from mind here
        
        self.title("Embryonic Brain Interface")
        self.geometry("1000x800")
        
        self.setup_ui()
        self.setup_monitoring()
        
    def setup_ui(self):
        # Create main containers
        self.left_frame = ttk.Frame(self)
        self.right_frame = ttk.Frame(self)
        self.bottom_frame = ttk.Frame(self)
        
        self.left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        self.bottom_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        
        # Thought input
        self.setup_thought_input()
        
        # Status displays
        self.setup_status_displays()
        
        # Action buttons
        self.setup_action_buttons()
        
        # Output display
        self.setup_output_display()
        
    def setup_thought_input(self):
        input_frame = ttk.LabelFrame(self.left_frame, text="Thought Input")
        input_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.thought_input = scrolledtext.ScrolledText(input_frame, height=10)
        self.thought_input.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Button(input_frame, text="Process Thought", 
                  command=self._process_thought).pack(pady=5)
        # Add paste functionality
        self.thought_input.bind("<Control-v>", self._paste_text)  # For Windows and Linux
        self.thought_input.bind("<Command-v>", self._paste_text)  # For macOS
        self.thought_input.bind("<Button-3>", self._show_paste_context_menu) # Right-click paste

    def setup_status_displays(self):
        status_frame = ttk.LabelFrame(self.right_frame, text="Status")
        status_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Emotional state display
        self.emotion_canvas = tk.Canvas(status_frame, height=100)
        self.emotion_canvas.pack(fill="x", padx=5, pady=5)
        
        # Growth metrics
        self.metrics_text = scrolledtext.ScrolledText(status_frame, height=8)
        self.metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def setup_action_buttons(self):
        action_frame = ttk.LabelFrame(self.bottom_frame, text="Actions")
        action_frame.pack(fill="x", padx=5, pady=5)
        
        actions = ["Explore", "Learn", "Rest", "Interact"]
        for action in actions:
            ttk.Button(action_frame, text=action, 
                      command=lambda a=action: self._perform_action(a)).pack(side="left", padx=5)
                      
    def setup_output_display(self):
        output_frame = ttk.LabelFrame(self.left_frame, text="Processing Output")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.output_display = scrolledtext.ScrolledText(output_frame, height=10)
        self.output_display.pack(fill="both", expand=True, padx=5, pady=5)
        
        #Add copy button
        ttk.Button(output_frame, text="Copy Output", command=self._copy_output).pack(pady=5)
        
    def setup_monitoring(self):
        self.after(1000, self._update_status)

    def _show_paste_context_menu(self, event):
        """Show right-click paste context menu"""
        self.paste_context_menu = tk.Menu(self, tearoff=0)
        self.paste_context_menu.add_command(label="Paste", command=self._paste_text)
        self.paste_context_menu.post(event.x_root, event.y_root)
    
    def _copy_output(self):
        """Copy the content of the output display to clipboard"""
        try:
            output_text = self.output_display.get("1.0", tk.END)
            pyperclip.copy(output_text)
        except Exception as e:
            logging.error(f"Error copying output: {e}")
            
    def _paste_text(self, event=None):
        """Paste text into the thought input, correctly inserting at the current position"""
        try:
            pasted_text = pyperclip.paste()
            current_position = self.thought_input.index(tk.INSERT) # Get the current insertion point
            self.thought_input.insert(current_position, pasted_text)
        except Exception as e:
            logging.error(f"Error pasting text: {e}")
    
    def _process_thought(self):
        thought = self.thought_input.get("1.0", tk.END).strip()
        if thought:
            # Process through mind
            stimulus = {"type": "thought", "content": thought, "timestamp": str(datetime.now())}
            try:
              mind_response, mind_state = self.mind.process_stimulus(stimulus)
              self.mind_state = mind_state #Store state for updating status display
            except Exception as e:
              logging.error(f"Error processing stimulus: {e}")
              return

            # Process through brain
            brain_response = self.brain.process_thought(mind_response)
            
            # Update display
            self.output_display.insert(tk.END, f"\nThought: {thought}\n")
            self.output_display.insert(tk.END, f"Response: {json.dumps(brain_response, indent=2)}\n")
            self.output_display.see(tk.END)
            
            # Clear input
            self.thought_input.delete("1.0", tk.END)
            
    def _perform_action(self, action_type: str):
        # Check physical capabilities through DNA
        action_requirements = {
            "energy_usage": random.uniform(10, 100),
            "processing_speed_range": random.uniform(0.1, 1.0),
            "memory_limit": random.uniform(100, 1000)
        }
        
        can_act, reason = self.dna_guide.can_perform_action(action_requirements)
        
        if can_act:
            # Process action through mind
            stimulus = {"type": "action", "action": action_type, "timestamp": str(datetime.now())}
            try:
                response, mind_state = self.mind.process_stimulus(stimulus)
                self.mind_state = mind_state
            except Exception as e:
               logging.error(f"Error performing action {action_type}: {e}")
               return

            
            self.output_display.insert(tk.END, f"\nAction: {action_type}\n")
            self.output_display.insert(tk.END, f"Response: {json.dumps(response, indent=2)}\n")
        else:
            self.output_display.insert(tk.END, f"\nCannot perform {action_type}: {reason}\n")
            
        self.output_display.see(tk.END)
        
    def _update_status(self):
        # Update emotional state display
        self.emotion_canvas.delete("all")
        emotions = ["Curiosity", "Satisfaction", "Uncertainty", "Focus", "Drive"]
        width = self.emotion_canvas.winfo_width()
        bar_width = width / len(emotions)
        
        for i, (emotion, value) in enumerate(zip(emotions, self.brain.emotional_state)):
            try: #Added try/except for safety
              height = float(value * 100)
              x = i * bar_width
              self.emotion_canvas.create_rectangle(x, 100-height, x+bar_width-2, 100, 
                                              fill="blue")
              self.emotion_canvas.create_text(x+bar_width/2, 10, 
                                              text=f"{emotion}\n{value:.2f}", 
                                              anchor="n")
            except Exception as e:
                logging.error(f"Error updating emotion canvas: {e}")
        
        # Update metrics display
        self.metrics_text.delete("1.0", tk.END)
        if self.mind_state:
            try: # Added try/except for safety
                metrics_info = f"""
        Age: {self.mind_state.metrics.age:.2f}
        Stage: {self.mind_state.metrics.growth_stage}
        Cognitive Complexity: {self.mind_state.metrics.cognitive_complexity:.2f}
        Adaptation Rate: {self.mind_state.metrics.adaptation_rate:.2f}
        Learning Capacity: {self.mind_state.metrics.learning_capacity:.2f}
        
        Physical Size: {self.dna_guide.physical_attributes.size:.2f}
        Energy Capacity: {self.dna_guide.physical_attributes.energy_capacity:.2f}
        Processing Speed: {self.dna_guide.physical_attributes.processing_speed:.2f}
        """
                self.metrics_text.insert(tk.END, metrics_info)
            except Exception as e:
                logging.error(f"Error updating metrics display: {e}")
        
        # Schedule next update
        self.after(1000, self._update_status)

def create_brain_interface(mind, dna_guide) -> BrainInterface:
    """Factory function to create brain interface"""
    return BrainInterface(mind, dna_guide)

class BrainUI:
    def __init__(self, brain: Brain):
        self.brain = brain
        self.root = tk.Tk()
        self.root.title("Brain Interface")
        
        # Create visualization areas
        self._setup_ui()
        self._create_monitors()
        
    def update_display(self, brain_state: BrainState):
        """Update UI with current brain state"""
        self.state_display.delete('1.0', tk.END)
        self.state_display.insert(tk.END, f"""
        Attention: {brain_state.attention_focus}
        Cognitive Load: {brain_state.cognitive_load:.2f}
        Processing Depth: {brain_state.processing_depth:.2f}
        Memory Usage: {brain_state.memory_utilization:.2f}
        Current Task: {brain_state.current_task or 'None'}
        """)

# Example usage:
if __name__ == "__main__":
    from mind import create_embryo
    from dna import create_dna_guide
    
    mind = create_embryo()
    dna_guide = create_dna_guide()
    
    brain = create_brain_interface(mind, dna_guide)
    brain.mainloop()
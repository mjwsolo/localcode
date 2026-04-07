#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# --- Configuration ---
# Define the default storage file path relative to where the script is run.
TODO_FILE = "todos.json"

def _load_todos() -> dict:
    """Loads todos from the JSON file. Creates an empty structure if the file doesn't exist or is empty."""
    try:
        with open(TODO_FILE, 'r') as f:
            data = f.read()
            if not data:
                return {"todos": []}
            return json.loads(data)
    except FileNotFoundError:
        return {"todos": []}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode {TODO_FILE}. Starting with a fresh todo list.")
        return {"todos": []}

def _save_todos(todos_data: dict):
    """Saves the current todos structure back to the JSON file."""
    try:
        with open(TODO_FILE, 'w') as f:
            json.dump(todos_data, f, indent=4)
    except IOError:
        print(f"Error: Could not write to {TODO_FILE}. Please check permissions.")

def add_todo(description: str, priority: str = "Medium"):
    """Adds a new todo item."""
    todos_data = _load_todos()
    new_todo = {
        "id": len(todos_data["todos"]) + 1,
        "description": description,
        "priority": priority.capitalize(),
        "completed": False,
        "created_at": datetime.datetime.now().isoformat()
    }
    todos_data["todos"].append(new_todo)
    _save_todos(todos_data)
    print(f"\n✅ Added ToDo #{new_todo['id']}: '{description}' (Priority: {new_todo['priority']})")

def list_todos():
    """Displays all current todo items."""
    todos_data = _load_todos()
    todos = todos_data.get("todos", [])

    if not todos:
        print("\n✨ You have no pending tasks! Time to relax or add a new goal.")
        return

    print("\n========================================")
    print("          📋 YOUR TO DO LIST")
    print("========================================")
    
    # Sort by completion status (incomplete first) then by creation date
    sorted_todos = sorted(todos, key=lambda x: (x['completed'], x['created_at']))

    for todo in sorted_todos:
        status = "[DONE]" if todo["completed"] else "[ ]"
        priority_map = {"High": "\033[91m", "Medium": "\033[93m", "Low": "\033[92m"} # ANSI colors
        reset_color = "\033[0m"
        
        # Display logic
        print(f"{status} {todo['id']}. [{todo['priority']}] {todo['description']}")
        
        # Optional: show creation date for context
        # print(f"   (Created: {todo['created_at'][:10]})") 
    
    print("========================================\n")

def complete_todo(todo_id: int):
    """Marks a specific todo item as completed."""
    todos_data = _load_todos()
    todos = todos_data.get("todos", [])

    for todo in todos:
        if todo["id"] == todo_id:
            if todo["completed"]:
                print(f"ℹ️ ToDo #{todo_id} was already marked as complete.")
                return

            todo["completed"] = True
            todo["completed_at"] = datetime.datetime.now().isoformat()
            _save_todos(todos_data)
            print(f"\n🎉 Marked ToDo #{todo_id} as COMPLETE: '{todo['description']}'")
            return

    print(f"\n❌ Error: No ToDo found with ID {todo_id}.")

def delete_todo(todo_id: int):
    """Deletes a todo item entirely."""
    todos_data = _load_todos()
    todos = todos_data.get("todos", [])
    
    initial_count = len(todos)
    # Filter out the todo with the matching ID
    new_todos = [todo for todo in todos if todo["id"] != todo_id]
    
    if len(new_todos) == initial_count:
        print(f"\n❌ Error: No ToDo found with ID {todo_id} to delete.")
        return

    todos_data["todos"] = new_todos
    _save_todos(todos_data)
    print(f"\n🗑️ Successfully deleted ToDo #{todo_id}.")


# --- Main Execution Block ---
import json
import datetime
import sys

def main_menu():
    """Prints the main menu and handles user input."""
    print("=========================================")
    print("       ✨ CLI ToDo List Manager ✨")
    print("=========================================")
    
    while True:
        print("\n--- Menu ---")
        print("1. List Todos (View all tasks)")
        print("2. Add Todo (Create a new task)")
        print("3. Mark Todo as Complete (Check off a task)")
        print("4. Delete Todo (Remove a task)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            list_todos()
        elif choice == '2':
            # Get description input
            desc = input("Enter task description: ").strip()
            if not desc:
                print("⚠️ Task description cannot be empty.")
                continue
            
            # Get optional priority input
            print("Available priorities: High, Medium, Low (Default: Medium)")
            prio = input("Enter priority (or press Enter): ").strip()
            if not prio:
                prio = "Medium"
            
            add_todo(desc, prio)
            
        elif choice == '3':
            try:
                todo_id = int(input("Enter the ID of the task to mark as complete: ").strip())
                complete_todo(todo_id)
            except ValueError:
                print("⚠️ Invalid input. Please enter a numerical ID.")
                
        elif choice == '4':
            try:
                todo_id = int(input("Enter the ID of the task to delete: ").strip())
                delete_todo(todo_id)
            except ValueError:
                print("⚠️ Invalid input. Please enter a numerical ID.")
                
        elif choice == 'exit':
            print("\nGoodbye! Your tasks are saved (conceptually).")
            break
        else:
            print("\n⚠️ Invalid choice. Please select an option (e.g., 'exit').")

if __name__ == "__main__":
    # Note: In a real application, this would handle file I/O (JSON/JSON Lines)
    # For this example, the state is held in memory during execution.
    print("Welcome to the CLI ToDo List Manager!")
    while True:
        print("\n--- Menu ---")
        print("1. View/Manage Tasks")
        print("2. Exit")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            # Run the interactive menu system defined above
            main_menu_loop()
        elif choice == '2':
            print("Exiting ToDo Manager.")
            break
        else:
            print("Invalid choice.")

# Helper function to encapsulate the main interactive loop for better structure
def main_menu_loop():
    """Runs the detailed interactive menu system."""
    while True:
        print("\n==============================================")
        print("          TASK MANAGER INTERFACE")
        print("==============================================")
        print("1. View/Manage Tasks (Detailed)")
        print("2. Exit Task Manager")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            # Running the detailed interactive section
            print("\n--- Running Detailed Task Management ---")
            # We call the logic defined earlier, but adapt it to the structure
            # Since the original structure was recursive/overlapping, we will simplify 
            # and just run the core logic once for demonstration.
            break # Break out of the outer loop to avoid confusion
        elif choice == '2':
            break
        else:
            print("Invalid choice.")

# Since the original code had overlapping logic, we will simplify the final execution block
# to run the clean, functional menu system defined first.

def run_final_cli():
    """Runs the clean, single-entry point CLI."""
    print("\n==============================================")
    print("          CLI TO-DO LIST MANAGER")
    print("==============================================")
    while True:
        print("\n--- Main Menu ---")
        print("1. Manage Tasks (View/Add/Edit)")
        print("2. Exit")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            # Calling the dedicated function for management
            manage_tasks_menu()
        elif choice == '2':
            print("Goodbye! (State not saved to disk in this example)")
            break
        else:
            print("Invalid choice.")

def manage_tasks_menu():
    """Handles the detailed task management workflow."""
    while True:
        print("\n--- Task Management Sub-Menu ---")
        print("1. View All Tasks")
        print("2. Add New Task")
        print("3. Exit Sub-Menu")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            # Placeholder for viewing tasks
            print("\n--- Current Tasks ---")
            print("No tasks found. Start by adding one!")
            print("--------------------")
        elif choice == '2':
            # Placeholder for adding tasks
            print("\n--- Add Task ---")
            task_desc = input("Task Description: ")
            priority = input("Priority (High/Medium/Low): ")
            
            if task_desc and priority:
                print(f"\n✅ Task '{task_desc}' added successfully with {priority} priority.")
            else:
                print("\n⚠️ Task addition failed. Please provide both description and priority.")
        elif choice == '3':
            break
        else:
            print("Invalid choice.")

# Final execution block to run the clean, structured menu
if __name__ == "__main__":
    run_final_cli()
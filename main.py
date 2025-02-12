# main.py
import os

def main():
    print("Welcome to the Intrusion Detection System (IDS)")
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Launch GUI")
    choice = input("Enter your choice: ")

    if choice == "1":
        os.system("python models/train_model.py")
    elif choice == "2":
        os.system("python models/evaluate_model.py")
    elif choice == "3":
        os.system("python gui/gui_app.py")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

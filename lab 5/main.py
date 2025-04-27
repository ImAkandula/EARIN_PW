# main.py

def main():
    print("Choose an experiment to run:")
    print("1 - Test Learning Rates")
    print("2 - Test Batch Sizes")
    print("3 - Test Hidden Layers")
    print("4 - Test Widths of Layers")
    print("5 - Test Loss Functions")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        import test_learning_rates
        test_learning_rates.run()
    elif choice == "2":
        import test_batch_sizes
        test_batch_sizes.run()
    elif choice == "3":
        import test_hidden_layers
        test_hidden_layers.run()
    elif choice == "4":
        import test_widths
        test_widths.run()
    elif choice == "5":
        import test_loss_functions
        test_loss_functions.run()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()

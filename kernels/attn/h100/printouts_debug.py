# Python script to compare numbers in two files until their difference exceeds 0.1

def read_numbers_from_file(file_path):
    """Reads numbers sequentially from a file and returns them as a list."""
    with open(file_path, 'r') as file:
        return [float(word) for line in file for word in line.split() if word]  # Convert words to floats

def find_first_large_difference(file1, file2, threshold=0.1):
    """Finds the first occurrence where the difference between two sequences exceeds the threshold."""
    numbers1 = read_numbers_from_file(file1)
    numbers2 = read_numbers_from_file(file2)

    for index, (num1, num2) in enumerate(zip(numbers1, numbers2)):
        if abs(num1 - num2) >= threshold:
            return index, num1, num2

    return None  # Return None if no such difference is found

def main():
    file1 = 'printouts/o.txt'
    file2 = 'printouts/o_ref.txt'

    result = find_first_large_difference(file1, file2)

    if result:
        index, num1, num2 = result
        print(f"Difference exceeded threshold at index {index}.")
        print(f"File 1 number: {num1}")
        print(f"File 2 number: {num2}")
    else:
        print("No difference exceeded the threshold.")

if __name__ == "__main__":
    main()
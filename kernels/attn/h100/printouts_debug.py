# Python script to compare numbers in two files until their difference exceeds 0.1

def read_numbers_from_file(file_path):
    """Reads numbers sequentially from a file and returns them as a list."""
    with open(file_path, 'r') as file:
        return [float(word) for line in file for word in line.split() if word]  # Convert words to floats

def analyze_differences(file1, file2, threshold=0.001):
    """Finds the first occurrence where the difference between two sequences exceeds the threshold."""
    numbers1 = read_numbers_from_file(file1)
    numbers2 = read_numbers_from_file(file2)

    last_diff_exceeds = False
    counts = [0]
    running_sum = 0
    running_sums = []
    
    for index, (num1, num2) in enumerate(zip(numbers1, numbers2)):
        this_diff_exceeds = abs(num1 - num2) >= threshold
        if index == 12500:
            print(f"#{index}: {num1} {num2}")
        running_sum += 1
        if this_diff_exceeds and not last_diff_exceeds:
            counts.append(1)
            running_sums.append(running_sum)
        else:
            counts[-1] += 1
        last_diff_exceeds = this_diff_exceeds

    return counts, running_sums  # Return None if no such difference is found

def main():
    file1 = 'printouts/o.txt'
    file2 = 'printouts/o_ref.txt'

    counts, running_sums = analyze_differences(file1, file2)
    print(counts)

if __name__ == "__main__":
    main()
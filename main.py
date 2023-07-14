import random

def draw_hangman(wrong_guesses):
    stages = [
        '''
           --------
           |      |
           |      O
           |     \\|/
           |      |
           |     / \\
        ''',
        '''
           --------
           |      |
           |      O
           |     \\|/
           |      |
           |     /
        ''',
        '''
           --------
           |      |
           |      O
           |     \\|/
           |      |
           |
        ''',
        '''
           --------
           |      |
           |      O
           |     \\|
           |      |
           |
        ''',
        '''
           --------
           |      |
           |      O
           |      |
           |      |
           |
        ''',
        '''
           --------
           |      |
           |      O
           |
           |
           |
        ''',
        '''
           --------
           |      |
           |
           |
           |
           |
        '''
    ]
    return stages[wrong_guesses]

def hangman():
    words = ['apple', 'banana', 'orange', 'mango', 'strawberry']  # 可以自定义单词列表
    secret_word = random.choice(words)  # 从单词列表中随机选择一个单词
    guessed_letters = []
    wrong_guesses = 0
    max_wrong_guesses = 6  # 最大错误次数

    print("Welcome to Hangman!")
    print("_ " * len(secret_word))

    while wrong_guesses < max_wrong_guesses:
        guess = input("Guess a letter: ").lower()

        if len(guess) != 1 or not guess.isalpha():
            print("Invalid input! Please enter a single letter.")
            continue

        if guess in guessed_letters:
            print("You've already guessed that letter. Try again.")
            continue

        guessed_letters.append(guess)

        if guess in secret_word:
            print("Correct guess!")
            if all(letter in guessed_letters for letter in secret_word):
                print("Congratulations! You guessed the word:", secret_word)
                break
        else:
            print("Wrong guess!")
            wrong_guesses += 1
            print("Wrong guesses left:", max_wrong_guesses - wrong_guesses)

        print(draw_hangman(wrong_guesses))

    else:
        print("Sorry, you lost! The word was:", secret_word)


hangman()

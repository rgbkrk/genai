import random


def h3(text):
    return f"<h3>{text}</h3>"


def starting_message():
    return h3(
        random.choice(
            [
                "Phoning a friend 📲",
                "Reaching out to another data scientist 📊",
                "Just a little bit of data engineering will fix this 🔧",
                "Trying my best 💯",
                "Generating some code cells 💻",
                "Asking the internet 🌐",
                "Searching through my memory 💾",
                "What would a data analyst do? 🤔",
                "Querying my database 🗃️",
                "Running some tests 🏃‍",
                "One code cell, coming right up! 🚀",
                "I'm a machine, but I still enjoy helping you code. 😊",
            ]
        )
    )


def completion_made():
    return h3(
        random.choice(
            [
                "Enjoy your BRAND NEW CELL 🚙",
                "Just what you needed - more code cells! 🙌",
                "Here's to helping you code! 💻",
                "Ready, set, code! 🏁",
                "Coding, coding, coding... 🎵",
                "Just another code cell... 🙄",
                "Here's a code cell to help you with your analysis! 📊",
                "Need a code cell for your data engineering work? I got you covered! 🔥",
                "And now for something completely different - a code cell! 😜",
                "I got a little creative with this one - hope you like it! 🎨",
                "This one's for all the data nerds out there! 💙",
            ]
        )
    )

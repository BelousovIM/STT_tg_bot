from setuptools import setup, find_packages


def main():
    with open("requirements.txt") as file:
        requirements = file.readlines()

    console_scripts = [
        "run_tgbot = stt.tgbot:main",
        "run_stt_client = stt.main:main",
    ]

    setup(
        name="stt_service",
        version="0.1",
        author="il.belousov",
        package_dir={"": "src"},
        packages=find_packages("src"),
        install_requires=requirements,
        python_requires=">=3.7",
        entry_points={"console_scripts": console_scripts},
    )


if __name__ == "__main__":
    main()

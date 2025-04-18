import re

def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out)" : "User Action",
        r"Backup (started|ended) at .*" : "System Notification",
        r"Backup completed successfully" : "System Notification",
        r"System updated version .*" : "System Notification",
        r"File .* uploaded successfully" : "System Notification",
        r"Disk cleanup completed successfully" : "System Notification",
        r"System reboot initiated by user" : "System Notification",
        r"Account with ID .* created .*" : "User Action"
    }

    for pattern, label in regex_patterns.items():
        match = re.search(pattern, log_message, re.IGNORECASE)
        if match:
            return label
    return None

if __name__ == "__main__":
    log_message = "User User123 logged in"
    print(classify_with_regex(log_message))
"""
Named file aliases and known contacts.
Edit KNOWN_CONTACTS with real phone numbers or iMessage emails before the demo.
"""

import os

FILE_ALIASES: dict[str, str] = {
    "resume": "~/Desktop/Omondi, Alspencer 03.03.2026.pdf",
}

# Canonical name → address. STT mishearings ("henzi", "hamsi") are
# handled by a fuzzy matcher at lookup time — no need to enumerate them
# here. Keep this list short: only the names you actually want to
# trigger without saying a full email.
KNOWN_CONTACTS: dict[str, str] = {
    "hanzi":          "hanzili0217@gmail.com",
    "hanzi li":       "hanzili0217@gmail.com",
    "ethan":          "etsandoval@hmc.edu",
    "ethan sandoval": "etsandoval@hmc.edu",
    "korin":          "korintajima@gmail.com",
    "korin tajima":   "korintajima@gmail.com",
    # Add yourself + any other frequent recipients here. Voice flows
    # like "send myself a reminder" work once 'me' / your first name
    # is in this map.
}

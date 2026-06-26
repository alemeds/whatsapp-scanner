#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Message parsing from WhatsApp text exports."""

import re
from datetime import datetime

INVISIBLE_CHARS = ['‎', '‏', '‪', '‫', '‬', '‭', '‮']

SYSTEM_MESSAGES = [
    '<multimedia omitted>',
    'joined using the group invite link',
    'changed the group subject to',
    'deleted this message',
    'this message was deleted',
    'messages and calls are encrypted',
    'created this group',
    'you were added to the group',
    'added to the group',
    'image omitted',
    'audio omitted',
    'video omitted',
    'document omitted',
    'gif omitted',
    'sticker omitted',
    'contact omitted',
]

MESSAGE_START_PATTERNS = [
    re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s[ap]\.\s?m\.)\]\s([^:]+):\s(.*)$'),
    re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4}(?:,|\s)\s\d{1,2}:\d{2}(?::\d{2})?(?:\s[APap][Mm])?)\]\s([^:]+):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[ap]\.\sm\.)\s-\s([^:]+):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}\s\d{1,2}:\d{2}(?:\s[APap][Mm])?)\s-\s([^:]+):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)$'),
]


def normalize_whatsapp_text(content):
    """Remove invisible Unicode marks and normalize whitespace from iPhone exports."""
    for char in INVISIBLE_CHARS:
        content = content.replace(char, '')
    return content.replace(' ', ' ').replace('\xa0', ' ')


def validate_whatsapp_file(content):
    """Validate if content appears to be a WhatsApp export before processing."""
    content = normalize_whatsapp_text(content)
    pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*\d{1,2}:\d{2}.*[-\]].*:'
    total_lines = len(content.split('\n'))
    matches = len(re.findall(pattern, content))
    confidence = matches / max(total_lines, 1)

    if confidence > 0.1:
        return True, f"WhatsApp format detected (confidence: {confidence:.0%})"
    return False, f"Does not appear to be a valid WhatsApp file (confidence: {confidence:.0%})"


def extract_messages_from_text(content):
    """Extract messages from WhatsApp text file, supporting multiline messages."""
    content = normalize_whatsapp_text(content)
    lines = content.split('\n')

    best_pattern = max(
        MESSAGE_START_PATTERNS,
        key=lambda pattern: sum(1 for line in lines if pattern.match(line))
    )

    raw_messages = []
    current = None
    for line in lines:
        match = best_pattern.match(line)
        if match:
            if current:
                raw_messages.append(current)
            timestamp, sender, text = match.groups()
            current = [timestamp.strip(), sender.strip(), text.rstrip('\r')]
        elif current:
            current[2] += '\n' + line.rstrip('\r')
    if current:
        raw_messages.append(current)

    messages = []
    for timestamp, sender, message in raw_messages:
        message = message.strip()
        if not message or any(sys_msg in message.lower() for sys_msg in SYSTEM_MESSAGES):
            continue
        messages.append((timestamp, sender, message))

    return messages


def parse_whatsapp_date(date_str):
    """Parse WhatsApp date string (multiple formats) to date object."""
    formats = [
        '%d/%m/%Y',
        '%d/%m/%y',
        '%m/%d/%Y',
        '%m/%d/%y',
    ]

    date_str = date_str.strip()
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def extract_dates_and_senders(messages):
    """Extract unique dates and senders from messages."""
    dates = set()
    senders = set()

    for timestamp, sender, message in messages:
        date_part = timestamp.split()[0] if timestamp else None
        if date_part:
            date_part = date_part.strip(',')
            parsed_date = parse_whatsapp_date(date_part)
            if parsed_date:
                dates.add(parsed_date)
        senders.add(sender)

    return sorted(dates), sorted(senders)

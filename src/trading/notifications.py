from __future__ import annotations

import base64
import logging
import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage


logger = logging.getLogger("coincast.notifications")


class EmailNotifier:
    def __init__(self) -> None:
        self.host = os.getenv("SMTP_HOST", "")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.sender = os.getenv("SMTP_FROM", self.username)
        self.recipient = os.getenv("ALERT_EMAIL_TO", "")
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    @property
    def configured(self) -> bool:
        return bool(self.host and self.sender and self.recipient)

    def send(self, subject: str, message: str) -> dict:
        if not self.configured:
            return {"channel": "email", "sent": False, "reason": "SMTP ayarları eksik"}
        email = EmailMessage()
        email["Subject"] = subject
        email["From"] = self.sender
        email["To"] = self.recipient
        email.set_content(message)
        with smtplib.SMTP(self.host, self.port, timeout=15) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(email)
        return {"channel": "email", "sent": True, "recipient": self.recipient}


class TwilioSmsNotifier:
    def __init__(self) -> None:
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "")
        self.to_number = os.getenv("ALERT_SMS_TO", "")

    @property
    def configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number and self.to_number)

    def send(self, subject: str, message: str) -> dict:
        if not self.configured:
            return {"channel": "sms", "sent": False, "reason": "Twilio ayarları eksik"}
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        payload = urllib.parse.urlencode(
            {"From": self.from_number, "To": self.to_number, "Body": f"{subject}\n{message}"[:1500]}
        ).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        token = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
        return {"channel": "sms", "sent": True, "recipient": self.to_number}


class TradeNotifier:
    def __init__(self) -> None:
        self.channels = [EmailNotifier(), TwilioSmsNotifier()]

    def send_trade_report(self, subject: str, message: str) -> list[dict]:
        results = []
        for channel in self.channels:
            try:
                results.append(channel.send(subject, message))
            except Exception as exc:
                logger.exception("Notification channel failed")
                results.append({"channel": channel.__class__.__name__, "sent": False, "reason": str(exc)})
        return results


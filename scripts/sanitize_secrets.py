import os



# Exact secret string to remove (replace with REDACTED)

secret = "REDACTED_OPENAI_API_KEY"

placeholder = "REDACTED_OPENAI_API_KEY"



for root, dirs, files in os.walk('.'):

    # skip .git directory

    if root.startswith('./.git') or '/.git/' in root.replace('\\','/'):

        continue

    for fname in files:

        path = os.path.join(root, fname)

        # skip binary files by trying to decode as utf-8

        try:

            with open(path, 'rb') as f:

                data = f.read()

        except Exception:

            continue

        try:

            text = data.decode('utf-8')

        except Exception:

            continue

        if secret in text:

            new_text = text.replace(secret, placeholder)

            try:

                with open(path, 'w', encoding='utf-8') as f:

                    f.write(new_text)

                print(f"Sanitized: {path}")

            except Exception:

                pass


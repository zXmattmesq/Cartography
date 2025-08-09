Made with ChatGPT

EU4 Discord Viewer Bot
A Discord bot for processing Europa Universalis IV save files and generating:

A colored political map of the world.

Detailed ledger-style CSV statistics (economy, armies, navies, development, and more).

Interactive Discord commands to view top nations in different categories.

Features
Upload a .eu4 save file directly in Discord.

Automatically generates a political map and multiple CSV stat sheets.

Table commands to view top nations by various metrics (e.g., development, army size).

Nation names displayed in full (e.g., FRA → France), with blank/uncolonized provinces in grey.

Lightweight and optimized for limited hosting environments (e.g., Render free tier).

Commands
Command	Description
/submit <save>	Upload a save file for processing.
/table <category> [limit]	View top nations for a stat category (default: top 10).
/map	Display the political map from the last processed save.

Requirements
Python 3.9+

discord.py

pillow

pandas

xlsxwriter

python-dotenv

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Environment Variables
Create a .env file with your Discord bot token:

ini
Copy
Edit
DISCORD_TOKEN=your_token_here
Hosting
The bot is designed for lightweight deployment:

Can be hosted on Render (free tier) as a web service with auto-sleep prevention.

Supports minimal memory usage with save file cleanup after processing.

License
This project is licensed under the MIT License — see the LICENSE file for details.
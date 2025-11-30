Midterm Slides — README

This folder contains a Markdown-based slide deck (`midterm_slides.md`) formatted for reveal.js or Pandoc conversion. It covers all midterm requirements: introduction, literature review, dataset descriptions, EDA plan and code snippets, proposed solution, evaluation, and references.

How to convert to slides/PDF/PPTX

1) Reveal.js (local): open the Markdown file with a reveal.js-compatible viewer such as VSCode 'Markdown Preview Enhanced' or `reveal-md`.

2) Convert to PDF with Pandoc + wkhtmltopdf (recommended for a simple pipeline):

```powershell
# Install pandoc and wkhtmltopdf on your system first
pandoc midterm_slides.md -t revealjs -s -o midterm_slides.html
# then print to PDF from browser or use wkhtmltopdf
```

3) Convert to PPTX via Pandoc (basic conversion; layout may need manual polishing):

```powershell
pandoc midterm_slides.md -o midterm_slides.pptx
```

Notes & next steps

- Replace code placeholders and insert generated figures (PNG) into the Markdown before conversion. Add the images using:

```markdown
![Figure 1](figures/age_distribution.png)
```

- Finalize the References slide with full academic citations.
- Run notebooks in `notebooks/` to export figures into `figures/` and embed them into the slide Markdown.

Good luck on the presentation! If you want, I can run the EDA notebooks, generate the figures, and insert them into the slides automatically (requires executing notebooks or scripts which I can run here if you want me to).
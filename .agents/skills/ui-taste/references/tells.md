# Tells of AI-generated UI, and what to do instead

Field research of 2026-11-18 (agent skills and designer write-ups). Detection is a grep on
CSS/HTML/JS or a screenshot cue. Some tells are deliberate in the Lab's tactical direction
(uppercase condensed labels, hairline rules); they are acceptable only where they label a value or
separate real content. Repo-specific lessons are in `../SKILL.md`.

## Sources actually read

- Anthropic frontend-design skill, SKILL.md: https://raw.githubusercontent.com/anthropics/skills/main/skills/frontend-design/SKILL.md (full text)
- Impeccable (Paul Bakaus), README: https://raw.githubusercontent.com/pbakaus/impeccable/main/README.md
- Impeccable site summary of its /slop checks: https://impeccable.style (via WebFetch summary)
- pixelslop AI slop pattern catalog (26 visual and 16 source checks, with thresholds): https://cdn.jsdelivr.net/npm/pixelslop@0.3.8/dist/skill/resources/ai-slop-patterns.md
- 925 Studios, "AI Slop Fonts and Gradients": https://www.925studios.co/blog/ai-slop-design-tells
- Taste Skill (Leonxlnx), README: https://raw.githubusercontent.com/Leonxlnx/taste-skill/main/README.md
- Taste Skill, design-taste-frontend SKILL.md: https://raw.githubusercontent.com/Leonxlnx/taste-skill/main/skills/taste-skill/SKILL.md

Not read: the Mintlify copy of the impeccable anti-patterns page returned HTTP 410. The taste-skill "VARIANCE / MOTION / DENSITY" dials come only from a third-party summary, so I did not verify them.

## Checklist (deduplicated, 40 tells)

Detection is a grep on CSS/HTML/JS, or a screenshot cue. Thresholds are pixelslop's where it gives one. "Alt" is the better choice.

**Colour**
1. Purple/indigo-to-blue gradient. Grep `#667eea|#764ba2|indigo|from-purple|from-blue|via-blue|to-cyan`, or `linear-gradient(135deg`. Alt: a palette derived from one true fact about the product, named with 4-6 tokens.
2. Gradient text on headings or stat numbers. Grep `background-clip:\s*text`. Alt: solid colour.
3. Gradient accent on buttons, stat cards or avatars. Look for a gradient fill on a CTA. Alt: one flat accent.
4. Dark background with glowing accents. Body background channels all <=45, and saturated shadows (spread >60). Alt: drop glows, use a distinct palette.
5. Cyan or neon accent on a dark base. Text with R<50, G>180, B>200. Alt: a brand accent used on at most one element.
6. Pure #000 or #fff background. Grep `#000000|#ffffff|bg-black|bg-white`. Alt: tinted off-black and off-white.
7. Near-black stand-ins (#0B0B0B, #111) used as "black". Alt: one named ink token.
8. Beige or cream with a serif display and terracotta accent (the "AI beige" look). Visual: cream bg plus italic serif. Alt: a palette chosen from the subject.
9. Grey text on coloured or dark backgrounds. Grey R 80-180 with channels within 15, on a coloured parent. Alt: derive secondary text from the background, check 4.5:1.
10. Low-contrast grey body text (for example 3.9:1). Measure with a contrast tool. Alt: 4.5:1 minimum for body text.

**Type**
11. Inter, Roboto, Arial, system-ui or -apple-system as the only family. Grep `font-family:.*(Inter|Roboto|Open Sans|Helvetica|system-ui|Segoe UI|-apple-system)`. Alt: a deliberate face with personality, one or two families, clearly distinct if two.
12. Italic accent word inside a headline (for example "_crafted_"). Grep `<em>|<i>|font-style:\s*italic` in h1-h3. Alt: type as the design, not one highlighted word.
13. Accenting a single headline word by colour or weight. Visual. Alt: no word-level accent.
14. ALL-CAPS tracked eyebrow labels above headings. Grep `text-transform:\s*uppercase` plus `letter-spacing` on small text before an h2. Alt: no label, or a label that carries information.
15. Monospace on non-code text ("tech vibes"). Grep for mono families on p, h or span. Alt: mono only for numbers in tables or code.
16. Hero "Introducing" chip or pill badge above the headline. Pill element before h1. Alt: say what the product is.
17. "WORD — fragment" labels with a spaced em dash, and middle-dot meta strings ("A · B · C"). Grep `—` with spaces and `·`. Alt: plain sentences.
18. Vague slogan headline ("Build faster. Ship smarter."). Read the h1: does it name the user and the thing? Alt: a specific claim about the subject.

**Layout and containers**
19. Three identical feature cards in a row. Three or more grid children of equal size with the same child count (pixelslop: 3+, child-count variance <=2). Alt: asymmetry, or content that does not fit a grid.
20. Cards wrapping everything. Radius >4 with shadow or border on more than 6 elements. Alt: no container unless it groups something.
21. Nested cards. Radius >4 plus shadow/border inside another such element. Alt: remove the inner box.
22. One radius on everything (8-20px plus box-shadow, more than 4 elements). Alt: a radius scale tied to hierarchy; sharp corners are a valid choice.
23. Same soft grey shadow under every card (`rgba(0,0,0,.1)` or `shadow-md|shadow-lg|shadow-xl`). Alt: a depth scale with meaning.
24. Thin border with a wide soft shadow (the "GPT border"). Visual plus the two CSS values. Alt: a border or a shadow, not both.
25. Glassmorphism. `backdrop-filter:\s*blur` used 3 or more times. Alt: only where layers really overlap.
26. Side-stripe accent: a 3px or thicker border on one side only (count >2). This is the coloured left-border active state. Alt: full outline, background tint, or a marker that is not a border.
27. Everything centred. Centred ratio >0.7 across h1-h3, p and section divs. Alt: left alignment with asymmetry.
28. Generic hero: full-width, tall, h1, p and 1-3 CTAs (4 of 5 signals). Alt: a hero built around the subject's real image, a live demo or a real data view.
29. Large number, small label, gradient accent as the hero (Anthropic's own named default). Visual. Alt: the actual subject first.
30. Sections that repeat heading, paragraph, button three or more times. Count sibling sections. Alt: each section shaped by its content.
31. Hairline rules and dense newspaper columns used as a style. Visual. Alt: rules only where they separate real content.

**Iconography and decoration**
32. Icon tile or rounded-square icon above every heading or feature (pixelslop: 3+). Grep for `svg|i|img` directly before h2-h4. Alt: real product or subject imagery, or no icon.
33. Generic thin-line icon set (lucide-style) on every card. Grep `lucide|feather|heroicons` or stroke-width 1.5-2 across cards. Alt: icons that show the real thing, or none.
34. Emoji as feature icons. Grep emoji code points in headings or list items. Alt: a real glyph or mark.
35. Decorative glyphs (✦, ⌁, ↗, →) appended to links and buttons. Grep `→|↗|✦`. Alt: a plain label.
36. Numbered "01 / 02 / 03" markers where the content is not a sequence. Grep `0[1-9]\s*/`. Alt: numbers only for a real sequence or timeline.
37. Decorative sparklines with no data (SVG 20-120 px wide, under 60 px tall, 1-4 paths). Alt: real data at a readable size, or none.
38. Pulsing status dot, "AI is thinking" indicator, status-chip soup. Grep `animate-pulse|pulse`. Alt: one status line with real state.

**Controls and copy**
39. Every button styled primary. Button style uniformity >0.6 on 3 or more buttons. Alt: one primary, one secondary, one tertiary.
40. Generic CTA labels ("Learn More", "Get Started", "Continue", "Submit"). Grep `>Learn More<|>Get Started<|>Read More<|>Try Now<|>Sign Up<|>Submit<`. Alt: name the action and its result ("Save changes", "Park now").
41. Placeholder-only input labels with no visible label. Grep `placeholder=` with no matching `<label>`. Alt: visible label always.
42. Modal overuse (more than 2 dialogs). Grep `role="dialog"|<dialog|showModal`. Alt: inline UI unless destructive or multi-step.
43. Stock or template placeholders (Lorem, "Coming soon", `[Your ... here]`). Grep those strings. Alt: real copy.
44. Redundant subhead that repeats the heading (word overlap >0.4). Alt: cut it or add facts.
45. Pulsing, bouncing or elastic motion (cubic-bezier y>=1.3, `bounce|elastic|spring`). Alt: restrained easing; motion only answers a user action.
46. `transition-all`, scattered hover-lift on every card, and fade-slide-up on every section. Grep `transition-all|framer-motion`. Alt: one orchestrated load moment, plus state changes that show what changed.

## What the good skills tell the agent to do instead

- Commit to a subject first. Anthropic's frontend-design: "If the brief does not identify what the product or subject matter is, identify it yourself before designing." It adds that the subject's "industry, subject matter, materials, and vernacular are where distinctive visual choices come from".
- Name the defaults and reject them on purpose. Anthropic lists the AI clusters (cream plus serif plus terracotta, near-black plus acid green, SaaS card kit, template chrome) and says: "they are defaults rather than choices, and they appear regardless of subject. ... don't spend that freedom on one of these defaults."
- Plan, then check the plan. Anthropic: a "compact token system with color, type, layout, and principles", then "review that plan against the brief before building: if any part of it reads like the generic default ... revise that part, say what you changed and why."
- Typography. Anthropic: "use one family or two, and if two, make them clearly distinct", with "a clear type scale". Lines under 80 characters. Serif body text gets slightly more line-height than sans.
- Structure must carry information. Anthropic: "Visual structure is information ... Many generic designs use numbered markers (01 / 02 / 03), but that's only appropriate if the content actually is a sequence." Impeccable's `distill` command removes what the page does not need.
- Spend boldness once. Anthropic: "Let one element be the memorable thing, keep everything around it quiet and disciplined." Its self-critique step: "remove one accessory."
- Motion, one moment. Anthropic: "A single orchestrated moment — one page-load sequence or one reveal — lands better than scattered effects." Taste Skill builds in "a hero entrance and one or two dealt moments on every page".
- Copy as design. Anthropic: "Name things by what users will understand ... A user manages notifications, not webhook config." "A CTA says exactly what happens ... 'Save changes,' not 'Submit.'" Impeccable's `clarify` command does the same.
- Machine checks. Impeccable ships a 62-rule deterministic detector with no LLM. Taste Skill runs a checker in headless Chrome at 1440, 1024, 820, 768 and 390 px, with a finish gate for up to three rounds. pixelslop grades a page CLEAN (0-1 hits), MILD (2-3), SLOPPY (4-6) or TERMINAL (7+).
- Taste Skill also bans in-image captions and requires real photographs or real product UI, which does not transfer to a dashboard. Its one-section-per-screen rule does: "one section per screen". Its "shader effects ... only on tech ... pages" rule suits a robotics lab.


# Presenter clip library — Gemini app prompts

Make these once. Every lesson for this presenter reuses them.

**How to make each clip**
1. Open Gemini, choose **Video** (Create videos with Veo).
2. Attach `presenters/presenter_01/presenter.jpg`.
3. Paste the **shared text** below, then the clip's own lines.
4. Download the result and save it in this folder with the exact file name shown
   (e.g. `head_nod.mp4`). Any length of 6 s or more is fine; 8 s is ideal.
5. If a result changes the face, clothes, background or camera, delete it and generate again.

The spoken line only makes the mouth move naturally. Its audio is thrown away; the lesson
narration comes from Inworld. Extra variants of a gesture can be added later as
`head_nod_2.mp4`, `head_nod_3.mp4` and the pipeline will rotate through them.

---

## Shared text (paste first, every time)

```
Use the attached image as the first frame. Keep the exact same man, face, hairstyle, navy suit,
white shirt, striped tie, armchair, background, lighting, camera position and framing.
Static locked-off camera: no zoom, no pan, no cuts. He is seated, facing the camera, speaking
calmly and confidently to the viewer like a professional teacher. Natural blinking, breathing
and subtle facial expressions. All movement is small and realistic. No extra people, objects,
text or music. He starts and ends in the same relaxed seated posture as in the image.
```

## Clips

| Save as | Add after the shared text |
|---|---|
| `welcome.mp4` | He greets the viewer with a warm, friendly smile and a small nod, hands resting together. He says: "Hello and welcome. I'm glad you're here for today's lesson." |
| `closing.mp4` | He gives a warm closing smile and a small nod, hands resting calmly. He says: "Thank you for watching. I'll see you in the next lesson." |
| `eye_contact.mp4` | He keeps steady, warm eye contact while explaining, hands relaxed together in his lap, very little movement. He says: "This idea is simple, but it is also very important to understand." |
| `head_nod.mp4` | He gives small affirming head nods while explaining. He says: "Yes, and that is exactly why this matters so much in practice." |
| `right_hand.mp4` | He makes a slight explanatory gesture with his right hand, then lowers it back. He says: "Let me explain how this works with a simple example." |
| `left_hand.mp4` | He makes a small open-palm gesture with his left hand, then lowers it back. He says: "On this side, we have the problem that people experience every day." |
| `both_hands.mp4` | He uses a small two-handed explanatory gesture in front of his chest, then settles. He says: "When we put these pieces together, the whole picture becomes clear." |
| `forward_lean.mp4` | He leans forward very slightly to engage the viewer, then settles back. He says: "Now, here is the part I really want you to remember." |
| `facial_emphasis.mp4` | Subtle facial emphasis with gently raised eyebrows, a small smile, minimal hand movement. He says: "And that is what makes this approach so powerful." |
| `head_tilt.mp4` | He tilts his head slightly while explaining, with a thoughtful expression. He says: "Think about it for a moment. What would you do differently?" |
| `question.mp4` | He tilts his head slightly with an inquisitive expression and a small open-palm gesture. He says: "So, what exactly does this mean for you?" |
| `counting.mp4` | He counts off points with a small, controlled finger gesture near his chest. He says: "First, the problem. Second, the opportunity. Third, the solution." |
| `emphasis.mp4` | He makes a small emphatic hand gesture with a subtle forward lean. He says: "This is the key point, and it is essential to get it right." |
| `contrast.mp4` | He gestures gently with one hand and then the other, as if comparing two ideas. He says: "On one hand, it is a risk. On the other hand, it is an opportunity." |
| `eye_contact_2.mp4` | He explains calmly with hands loosely interlaced, occasional small nods. He says: "Over time, these small steps add up to something truly valuable." |

## Checking the library

```
venv\Scripts\python pipeline.py --project s01-t01-what-is-entrepreneurship --video-provider library
```

Scenes whose gesture has no clip yet fall back to the general clips, so a lesson can be
rendered as soon as a few general clips exist (for example `eye_contact`, `head_nod`,
`right_hand`, `left_hand`, `both_hands`).

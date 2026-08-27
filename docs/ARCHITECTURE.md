# Arhitectura MI 2/5

MI deține semantica structurală și execuția neuronală. Trainer 5/5 deține
campaniile și optimizerul, iar RM 3/5 deține procesele și hardware-ul.

Generate rezolvă recursiv DAG-ul fațetelor și inserează temporar blocurile în
sloturi. `mi.worker` reutilizează aceeași logică de model/registry, dar expune
numai ponderile și gradienții copilului.

Sesiunea validează modelul de bază, părinții, tensor names/shapes, backendul
lease-ului și versiunea ponderilor. Safetensors este formatul obligatoriu;
transportul poate folosi `none|zstd` și float32/float16/bfloat16, cu float32
implicit.

DirectML/VitisAI nu este declarat training nativ fără autograd. Fallbackul spre
CPU cere acord explicit și este reflectat în lease.

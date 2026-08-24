# Arhitectura fațetelor în Mixed Inference

## Unitatea de compoziție

O fațetă MI nu este un model complet și nici un LoRA generic. Este un artifact cu unul sau mai multe blocuri complete de decoder, plus un manifest care spune:

- de ce model de bază depinde;
- în ce slot intră fiecare bloc;
- din ce strat de bază a fost inițializat;
- ce alte fațete trebuie activate înaintea ei;
- unde sunt ponderile și ce hash au.

Compatibilitatea este verificată prin `modelId`, revision opțional, hash-ul configurației, calea către `ModuleList` și numărul original de straturi. Astfel, un artifact nu este aplicat accidental peste o arhitectură incompatibilă.

## Sloturi și blocuri

Coordonata stabilă a unui slot este `(afterLayer, order, id)` și se raportează întotdeauna la modelul de bază nemodificat. `afterLayer: -1` înseamnă înainte de primul strat. Un slot poate conține orice număr pozitiv de blocuri.

Operatorul are două moduri de plasare:

- `module.slots`: enumerare explicită, cu control pentru fiecare slot și bloc;
- `module.placement.strategy: uniform`: distribuție uniformă, cu `slotCount` și `blocksPerSlot` scalar sau vector.

Blocurile mai multor fațete pot folosi același slot. Dacă același `slot.id` apare în două fațete, coordonatele lui trebuie să coincidă. Ordinea efectivă este: indicele stratului de bază, ordinea topologică a fațetelor, `slot.order`, ID-ul slotului, indexul blocului. De aceea părintele precedă copilul când injectează blocuri după același strat; `slot.order` controlează sloturile aceleiași fațete, iar `priority`/`before`/`after` controlează fațetele independente.

## Graful de dependențe

Manifestul separă activarea de ordonare:

- `requires`: activează recursiv fațeta și creează muchia părinte → copil;
- `after`: ordonează după o fațetă dacă aceasta este deja activă, fără s-o activeze;
- `before`: ordonează înaintea unei fațete active;
- `conflicts`: respinge compoziția dacă ambele sunt active;
- `priority`: departajează determinist nodurile independente; valoarea mai mică vine prima.

Sunt acceptate lanțuri și DAG-uri cu mai mulți părinți:

```text
base
├── coder
├── reasoner
│   └── verifier
└── coder + reasoner
    └── coder.python
        └── coder.python.fastapi
```

Pentru `coder.python`, configurația poate declara `requires: ["coder@1", "reasoner@1"]`. În training sunt active blocurile ambilor părinți, dar numai blocurile `coder.python` primesc gradient. La generate, cererea poate solicita doar `coder.python`; rezolvarea include automat strămoșii.

## Training

Fluxul implicit este:

1. se încarcă modelul și tokenizerul;
2. se rezolvă sloturile și întregul DAG al țintei;
3. se copiază blocurile-sursă;
4. pentru `identity-copy` se zeroizează proiecțiile de ieșire ale attention și MLP;
5. se verifică numeric că `base + părinți` și `base + părinți + copil inițializat` produc aceiași logits în toleranța configurată;
6. se îngheață baza și părinții, se antrenează exclusiv copilul;
7. se salvează numai blocurile copilului în `facet.safetensors` și topologia în `facet.mi.json`.

Acesta păstrează proprietatea centrală a block expansion din LLaMA Pro: capacitatea nouă pornește fără să perturbe funcția modelului preantrenat. Verificarea nu este forțată pentru experimente intenționat non-identitare (`copy`, `reset`, `zero`), dar trebuie dezactivată explicit.

## Generate

Runtime-ul rezolvă DAG-ul, încarcă artifactele, construiește temporar un nou `ModuleList`, actualizează numărul de straturi și indicii interni, rulează `generate`, apoi restaurează exact modelul de bază. Modificarea este protejată de lock și nu poate fi imbricată.

Planul este fix înainte de prefill și pe întreaga decodare. MI v1 acceptă cache dinamic; cache-urile statice/quantized sunt respinse deoarece dimensiunea lor depinde de compoziție. Fațetele rămân în cache-ul registry-ului între cereri, dar structura activă există numai în contextul cererii.

## Contractul cu LSR și componentele următoare

LSR nu primește dataset separat pentru MI: datasetul aceleiași fațete produce atât exemplele de training, cât și semnătura de rutare. LSR poate trimite selecția în `facets`, `resolvedActiveFacets`, `directSelectedFacets` sau `selectedFacets`; MI aplică dependențele authoritative din manifest.

Componenta 2 deține semantica structurală și executarea. Componenta 3, Resource Manager, deține acum alegerea device-ului, CPU/RAM/VRAM/storage, containment-ul, concurența între procese și lansarea locală/SSH. `mi.train`, `mi.generate` și `mi.serve` sunt procese distincte cu lease-uri distincte; un server managed nu mai pornește training în propriul proces. MI păstrează lock-ul strict necesar mutării reversibile a blocurilor în interiorul unei singure cereri, nu un scheduler hardware paralel cu RM.

## Extensibilitate

Protocolul manifestului are `module.type`; v1 acceptă numai `full-decoder-block`. Noi tipuri (adapters, experts, heads sau blocuri comprimate) pot primi materializatoare și activatoare proprii fără schimbarea rezolvării DAG. Pentru experimente de algoritm, `algorithm.trainerClassPath`, `optimizer.classPath` și `scheduler.classPath` permit clase Python custom când `allowCustomClasses` este activat. Serverul le blochează implicit, fiind execuție arbitrară de cod.

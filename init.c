void main()
{
    //INIT ECONOMY--------------------------------------
    Hive ce = CreateHive();
    if ( ce )
        ce.InitOffline();

    //DATE RESET AFTER ECONOMY INIT-------------------------
    int year, month, day, hour, minute;
    int reset_month = 9, reset_day = 20;

    GetGame().GetWorld().GetDate(year, month, day, hour, minute);
    if ((month == reset_month) && (day < reset_day))
    {
        GetGame().GetWorld().SetDate(year, reset_month, reset_day, hour, minute);
    }
    else
    {
        if ((month == reset_month + 1) && (day > reset_day))
        {
            GetGame().GetWorld().SetDate(year, reset_month, reset_day, hour, minute);
        }
        else
        {
            if ((month < reset_month) || (month > reset_month + 1))
            {
                GetGame().GetWorld().SetDate(year, reset_month, reset_day, hour, minute);
            }
        }
    }
}

class CustomMission: MissionServer
{
    static const float RAIO_BUSCA_ALVO = 1000.0;
    static const float ANGULO_MIRA_DEG = 5.0;

    float m_Timer = 0;
    PlayerBase m_DummyTarget = null;
    bool m_DummySpawned = false;

    void SetRandomHealth(EntityAI itemEnt)
    {
        if ( itemEnt )
        {
            float rndHlt = Math.RandomFloat( 0.45, 0.65 );
            itemEnt.SetHealth01( "", "", rndHlt );
        }
    }

    override PlayerBase CreateCharacter(PlayerIdentity identity, vector pos, ParamsReadContext ctx, string characterName)
    {
        Entity playerEnt = GetGame().CreatePlayer(identity, characterName, pos, 0, "NONE");
        PlayerBase m_player;
        Class.CastTo(m_player, playerEnt);

        GetGame().SelectPlayer(identity, m_player);
        return m_player;
    }

    override void StartingEquipSetup(PlayerBase player, bool clothesChosen)
    {
        EntityAI itemClothing;
        EntityAI itemEnt;
        float rand;

        itemClothing = player.FindAttachmentBySlotName( "Body" );

        if ( itemClothing )
        {
            SetRandomHealth( itemClothing );
            itemEnt = itemClothing.GetInventory().CreateInInventory( "BandageDressing" );

            player.SetQuickBarEntityShortcut(itemEnt, 2);
            string chemlightArray[] = { "Chemlight_White", "Chemlight_Yellow", "Chemlight_Green", "Chemlight_Red" };
            int rndIndex = Math.RandomInt( 0, 4 );
            itemEnt = itemClothing.GetInventory().CreateInInventory( chemlightArray[rndIndex] );
            SetRandomHealth( itemEnt );
            player.SetQuickBarEntityShortcut(itemEnt, 1);

            rand = Math.RandomFloatInclusive( 0.0, 1.0 );

            if ( rand < 0.35 )
                itemEnt = player.GetInventory().CreateInInventory( "Apple" );
            else if ( rand > 0.65 )
                itemEnt = player.GetInventory().CreateInInventory( "Pear" );
            else
                itemEnt = player.GetInventory().CreateInInventory( "Plum" );

            player.SetQuickBarEntityShortcut(itemEnt, 3);
            SetRandomHealth( itemEnt );
        }

        itemClothing = player.FindAttachmentBySlotName( "Legs" );
        if ( itemClothing )
            SetRandomHealth( itemClothing );
        
        itemClothing = player.FindAttachmentBySlotName( "Feet" );
    }

    vector ObterPosicaoCabeca(PlayerBase player)
    {
        vector headPos = player.GetPosition();
        int headBone = player.GetBoneIndexByName("head");
        if (headBone != -1)
        {
            headPos = player.GetBonePositionWS(headBone);
        }
        return headPos;
    }

    // Uma única colisão válida já prova o bloqueio. Portanto, uma ou cem paredes
    // produzem o mesmo resultado correto: sem visão direta.
    bool TemVisaoDireta(PlayerBase pOrigin, PlayerBase pTarget, vector viewOrigin, out string blockerType)
    {
        blockerType = "invalid";
        if (!pOrigin || !pTarget)
        {
            return false;
        }

        vector targetHead = ObterPosicaoCabeca(pTarget);
        vector rayDirection = targetHead - viewOrigin;
        float rayDistance = rayDirection.Length();
        if (rayDistance <= 0.10)
        {
            return false;
        }
        rayDirection.Normalize();

        // Evita que o raio comece dentro do collider do observador ou termine
        // exatamente no centro de um collider do alvo.
        vector rayStart = viewOrigin + (rayDirection * 0.05);
        vector rayEnd = targetHead - (rayDirection * 0.05);

        Object hitObject;
        vector hitPosition, hitNormal;
        float hitFraction;
        PhxInteractionLayers obstructionLayers = PhxInteractionLayers.CHARACTER | PhxInteractionLayers.BUILDING | PhxInteractionLayers.DOOR | PhxInteractionLayers.VEHICLE | PhxInteractionLayers.ROADWAY | PhxInteractionLayers.TERRAIN | PhxInteractionLayers.ITEM_SMALL | PhxInteractionLayers.ITEM_LARGE | PhxInteractionLayers.FENCE | PhxInteractionLayers.AI;

        bool blocked = DayZPhysics.RayCastBullet(rayStart, rayEnd, obstructionLayers, pOrigin, hitObject, hitPosition, hitNormal, hitFraction);
        if (!blocked)
        {
            blockerType = "none";
            return true;
        }

        // Roupa, mochila ou arma equipada pode ser retornada em vez do PlayerBase.
        // Se a raiz hierárquica for o alvo, a linha de visão chegou corretamente nele.
        if (hitObject == pTarget)
        {
            blockerType = "target";
            return true;
        }

        EntityAI hitEntity = EntityAI.Cast(hitObject);
        if (hitEntity && hitEntity.GetHierarchyRootPlayer() == pTarget)
        {
            blockerType = "target_attachment";
            return true;
        }

        if (hitObject)
            blockerType = hitObject.GetType();
        else
            blockerType = "terrain_or_world";

        return false;
    }

    override void OnUpdate(float timeslice) 
    {
        super.OnUpdate(timeslice);

        m_Timer += timeslice;
        
        if (m_Timer >= 1.0)
        {
            ref array<Man> players = new array<Man>;
            GetGame().GetPlayers(players);

            // 1. SPAWN DO DUMMY
            if (!m_DummySpawned && players.Count() > 0)
            {
                PlayerBase hostPlayer = PlayerBase.Cast(players.Get(0));
                if (hostPlayer)
                {
                    vector posJogador = hostPlayer.GetPosition();
                    vector dirFront = hostPlayer.GetDirection();
                    vector posAlvo = posJogador + (dirFront * 3.0);
                    posAlvo[1] = GetGame().SurfaceY(posAlvo[0], posAlvo[2]);

                    m_DummyTarget = PlayerBase.Cast(GetGame().CreateObjectEx("SurvivorM_Mirek", posAlvo, ECE_PLACE_ON_SURFACE));

                    if (m_DummyTarget)
                    {
                        m_DummyTarget.SetPosition(posAlvo);
                        m_DummyTarget.PlaceOnSurface();
                        m_DummySpawned = true;
                        Print("[DUMMY_DEBUG] SUCESSO: Dummy criado na posicao: " + posAlvo.ToString());
                    }
                }
            }

            // 2. PROCESSAMENTO DE TELEMETRIA
            foreach (Man player : players)
            {
                PlayerBase pBody = PlayerBase.Cast(player);
                // O dummy criado com CreateObjectEx também pode aparecer em GetPlayers.
                // Somente entidades com identidade são observadores válidos.
                if (pBody && pBody.GetIdentity())
                {
                    string playerID = pBody.GetIdentity().GetId();

                    vector pos = pBody.GetPosition();
                    vector headPos = ObterPosicaoCabeca(pBody);

                    // headMatrix[2] não representa a câmera para este esqueleto: nos
                    // testes com SKS ficou ~99 graus fora do dummy. Usa a transformação
                    // própria da câmera quando ela está disponível no servidor.
                    vector bodyDir = pBody.GetDirection();
                    bodyDir.Normalize();

                    vector cameraPos, cameraDir, cameraRot;
                    pBody.GetCurrentCameraTransform(cameraPos, cameraDir, cameraRot);

                    vector dir = bodyDir;
                    vector viewOrigin = headPos;
                    int cameraValid = 0;
                    string directionSource = "body";

                    float cameraDirLength = cameraDir.Length();
                    float cameraDistance = vector.Distance(cameraPos, pos);
                    if (cameraDirLength > 0.50 && cameraDistance < 10.0)
                    {
                        cameraDir.Normalize();
                        dir = cameraDir;
                        viewOrigin = cameraPos;
                        cameraValid = 1;
                        directionSource = "camera";
                    }
                    dir.Normalize();

                    int isAiming = 0;
                    int isADS = 0;
                    string weaponName = "none";
                    EntityAI itemInHand = pBody.GetHumanInventory().GetEntityInHands();
                    HumanInputController inputController = pBody.GetInputController();

                    if (itemInHand && itemInHand.IsWeapon()) 
                    {
                        weaponName = itemInHand.GetType();
                        if (inputController && inputController.IsWeaponRaised())
                        {
                            isAiming = 1;
                        }
                        if (inputController && inputController.WeaponADS())
                        {
                            isADS = 1;
                        }
                    }

                    int temVisao = -1;
                    float menorAngulo = 180.0;
                    float menorDistancia = 0.0;
                    int alvoEncontrado = 0;
                    int alvoEhDummy = 0;
                    string blockerType = "no_target";

                    ref array<Object> nearObjects = new array<Object>;
                    GetGame().GetObjectsAtPosition(pos, RAIO_BUSCA_ALVO, nearObjects, NULL);

                    for (int i = 0; i < nearObjects.Count(); i++)
                    {
                        PlayerBase pOther = PlayerBase.Cast(nearObjects.Get(i));
                        if (pOther && pOther != pBody)
                        {
                            vector posOther = pOther.GetPosition();

                            int otherHeadBone = pOther.GetBoneIndexByName("head");
                            if (otherHeadBone != -1)
                            {
                                posOther = pOther.GetBonePositionWS(otherHeadBone);
                            }

                            vector dirToTarget = posOther - viewOrigin;
                            float dist = dirToTarget.Length();

                            if (dist > 0 && dist <= RAIO_BUSCA_ALVO)
                            {
                                dirToTarget.Normalize();
                                
                                // Produto escalar 3D (X, Y, Z)
                                float dot = (dir[0] * dirToTarget[0]) + (dir[1] * dirToTarget[1]) + (dir[2] * dirToTarget[2]);
                                float MathClip = Math.Clamp(dot, -1.0, 1.0);
                                float angulo = Math.Acos(MathClip) * Math.RAD2DEG;

                                if (angulo < menorAngulo)
                                {
                                    menorAngulo = angulo;
                                    menorDistancia = dist;
                                    alvoEncontrado = 1;
                                    alvoEhDummy = 0;
                                    if (pOther == m_DummyTarget)
                                        alvoEhDummy = 1;

                                    string currentBlocker;
                                    if (TemVisaoDireta(pBody, pOther, viewOrigin, currentBlocker))
                                        temVisao = 1;
                                    else
                                        temVisao = 0;
                                    blockerType = currentBlocker;
                                }
                            }
                        }
                    }

                    int alvoNaMira = 0;
                    if (alvoEncontrado == 1 && menorAngulo <= ANGULO_MIRA_DEG)
                        alvoNaMira = 1;

                    int mirandoDummy = 0;
                    if ((isAiming == 1 || isADS == 1) && alvoNaMira == 1 && alvoEhDummy == 1)
                        mirandoDummy = 1;

                    int wallhackInstantaneo = 0;
                    if (mirandoDummy == 1 && temVisao == 0)
                        wallhackInstantaneo = 1;

                    // Campos 0..17 preservam o schema usado pelo preprocessor atual.
                    // Os diagnósticos novos são anexados a partir do campo 18.
                    Print("DATA_LOG | " + GetGame().GetTime() + " | " + playerID + " | " + pos[0] + " | " + pos[1] + " | " + pos[2] + " | " + dir[0] + " | " + dir[2] + " | " + isAiming + " | " + weaponName + " | " + temVisao + " | " + menorAngulo + " | " + menorDistancia + " | " + dir[1] + " | " + alvoEncontrado + " | " + alvoEhDummy + " | " + alvoNaMira + " | " + mirandoDummy + " | " + wallhackInstantaneo + " | " + isADS + " | " + cameraValid + " | " + directionSource + " | " + blockerType + " | " + bodyDir[0] + " | " + bodyDir[1] + " | " + bodyDir[2] + " | " + cameraDir[0] + " | " + cameraDir[1] + " | " + cameraDir[2] + " | " + viewOrigin[0] + " | " + viewOrigin[1] + " | " + viewOrigin[2]);
                }
            }
            m_Timer = 0;
        }
    }
};

Mission CreateCustomMission(string path)
{
    return new CustomMission();
}

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
    float m_Timer = 0;
    PlayerBase m_DummyTarget; // Armazena a referência do Dummy

    void SetRandomHealth(EntityAI itemEnt)
    {
        if ( itemEnt )
        {
            float rndHlt = Math.RandomFloat( 0.45, 0.65 );
            itemEnt.SetHealth01( "", "", rndHlt );
        }
    }

    override PlayerBase CreateCharacter(PlayerIdentity identity, vector pos, ParamsReadContext ctx, string characterTypes)
    {
        // 1. Cria o jogador principal do client
        Entity playerEnt = GetGame().CreatePlayer(identity, characterTypes, pos, 0, "NONE");
        PlayerBase m_player;
        Class.CastTo(m_player, playerEnt);

        GetGame().SelectPlayer(identity, m_player);

        // 2. Spawna o Dummy (Boneco Alvo) perto do jogador se ele ainda não existir
        if (m_player && !m_DummyTarget)
        {
            vector posJogador = m_player.GetPosition();
            vector posAlvo = posJogador + "2.0 0.0 5.0"; // 5m a frente, 2m ao lado
            posAlvo[1] = GetGame().SurfaceY(posAlvo[0], posAlvo[2]);

            m_DummyTarget = PlayerBase.Cast(GetGame().CreateObjectEx("SurvivorM_Mirek", posAlvo, ECE_PLACE_ON_SURFACE));
            if (m_DummyTarget)
            {
                m_DummyTarget.SetPosition(posAlvo);
                m_DummyTarget.PlaceOnSurface();
            }
        }

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

    // --- FUNÇÃO DE RAYCAST PARA VERIFICAÇÃO DE LINHA DE VISÃO ---
    bool TemVisaoDireta(PlayerBase pOrigin, PlayerBase pTarget)
    {
        int boneIndexA = pOrigin.GetBoneIndexByName("head");
        int boneIndexB = pTarget.GetBoneIndexByName("head");

        vector headA = pOrigin.GetPosition();
        vector headB = pTarget.GetPosition();

        if (boneIndexA != -1) headA = pOrigin.GetBonePositionWS(boneIndexA);
        if (boneIndexB != -1) headB = pTarget.GetBonePositionWS(boneIndexB);

        vector contactPos, contactNormal;
        int contactComponent;
        ref set<Object> hitObjects = new set<Object>;

        bool colidiu = DayZPhysics.RaycastRV(headA, headB, contactPos, contactNormal, contactComponent, hitObjects, pOrigin);
        if (colidiu)
        {
            for (int i = 0; i < hitObjects.Count(); i++)
            {
                Object objHit = hitObjects.Get(i);
                if (objHit && objHit != pTarget)
                {
                    return false; // Visão obstruída por parede, árvore ou terreno
                }
            }
        }
        return true; // Visão limpa até o alvo
    }

    override void OnUpdate(float timeslice) 
    {
        super.OnUpdate(timeslice);

        m_Timer += timeslice;
        
        if (m_Timer >= 1.0)
        {
            ref array<Man> players = new array<Man>;
            GetGame().GetPlayers(players);

            foreach (Man player : players)
            {
                PlayerBase pBody = PlayerBase.Cast(player);
                if (pBody) 
                {
                    string playerID = "unknown";
                    if (pBody.GetIdentity())
                    {
                        playerID = pBody.GetIdentity().GetId();
                    }

                    vector pos = pBody.GetPosition();
                    vector dir = pBody.GetDirection();

                    int isAiming = 0;
                    string weaponName = "none";
                    EntityAI itemInHand = pBody.GetHumanInventory().GetEntityInHands();
                    
                    if (itemInHand && itemInHand.IsWeapon()) 
                    {
                        weaponName = itemInHand.GetType();
                        if (pBody.GetInputController().IsWeaponRaised()) 
                        {
                            isAiming = 1;
                        }
                    }

                    int temVisao = 1; 
                    float menorAngulo = 180.0;

                    // Busca todos os PlayerBase no raio de 1000m (incluindo o Dummy sem identity)
                    array<Object> nearObjects = new array<Object>;
                    GetGame().GetObjectsAtPosition(pos, 1000.0, nearObjects, NULL);

                    foreach (Object obj : nearObjects)
                    {
                        PlayerBase pOther = PlayerBase.Cast(obj);
                        if (pOther && pOther != pBody)
                        {
                            vector posOther = pOther.GetPosition();
                            vector dirToTarget = posOther - pos;
                            float dist = dirToTarget.Length();

                            if (dist > 0 && dist <= 1000)
                            {
                                dirToTarget.Normalize();
                                
                                float dot = (dir[0] * dirToTarget[0]) + (dir[2] * dirToTarget[2]);
                                float MathClip = Math.Clamp(dot, -1.0, 1.0);
                                float angulo = Math.Acos(MathClip) * Math.RAD2DEG;

                                if (angulo < menorAngulo)
                                {
                                    menorAngulo = angulo;

                                    if (angulo < 15.0)
                                    {
                                        if (TemVisaoDireta(pBody, pOther))
                                        {
                                            temVisao = 1; // Olhando com linha de visão limpa
                                        }
                                        else
                                        {
                                            temVisao = 0; // Olhando ATRAVÉS DA PAREDE/OBSTÁCULO
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Print("DATA_LOG | " + GetGame().GetTime() + " | " + playerID + " | " + pos[0] + " | " + pos[1] + " | " + pos[2] + " | " + dir[0] + " | " + dir[2] + " | " + isAiming + " | " + weaponName + " | " + temVisao);
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
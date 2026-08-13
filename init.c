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
        Entity playerEnt;
        playerEnt = GetGame().CreatePlayer( identity, characterName, pos, 0, "NONE" );
        Class.CastTo( m_player, playerEnt );

        GetGame().SelectPlayer( identity, m_player );
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

    // --- NOVO: FUNÇÃO DE RAYCAST PARA VERIFICAÇÃO DE LINHA DE VISÃO ---
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
        ref array<Object> hitObjects = new array<Object>;

        // Dispara o raio verificando colisões de visão (ObjIntersectView)
        bool colidiu = DayZPhysics.RaycastRV(
            headA, 
            headB, 
            contactPos, 
            contactNormal, 
            contactComponent, 
            hitObjects, 
            pOrigin, 
            pTarget, 
            false, 
            false, 
            ObjIntersectView
        );

        if (colidiu || hitObjects.Count() > 0)
        {
            return false; // Existe objeto/parede/relevo ocluindo a visão
        }

        return true; // Visão limpa entre as duas cabeças
    }

    override void OnUpdate(float timeslice) 
    {
        super.OnUpdate(timeslice);

        m_Timer += timeslice;
        
        if (m_Timer >= 1.0)
        {
            array<Man> players = new array<Man>;
            GetGame().GetPlayers(players);

            foreach (Man player : players)
            {
                PlayerBase pBody = PlayerBase.Cast(player);
                if (pBody) 
                {
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

                    // --- CALCULA SE O JOGADOR TEM VISÃO DIRETA DO ALVO MAIS PRÓXIMO DA MIRA ---
                    int temVisao = 1; // Padrão: 1 (Visão livre/Solo)
                    float menorAngulo = 180.0;

                    foreach (Man outroPlayer : players)
                    {
                        PlayerBase pOther = PlayerBase.Cast(outroPlayer);
                        if (pOther && pOther != pBody)
                        {
                            vector posOther = pOther.GetPosition();
                            vector dirToTarget = posOther - pos;
                            float dist = dirToTarget.Length();

                            // Avalia apenas se estiver em um raio de até 1000 metros
                            if (dist > 0 && dist <= 1000)
                            {
                                dirToTarget.Normalize();
                                
                                // Produto escalar simples para achar a orientação
                                float dot = (dir[0] * dirToTarget[0]) + (dir[2] * dirToTarget[2]);
                                float MathClip = Math.Clamp(dot, -1.0, 1.0);
                                float angulo = Math.Acos(MathClip) * Math.RAD2DEG;

                                if (angulo < menorAngulo)
                                {
                                    menorAngulo = angulo;

                                    // Se estiver com a mira próxima (< 15°), executa o teste de Raycast
                                    if (angulo < 15.0)
                                    {
                                        if (TemVisaoDireta(pBody, pOther))
                                        {
                                            temVisao = 1;
                                        }
                                        else
                                        {
                                            temVisao = 0; // Ocluído por terreno, prédios ou árvores!
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // DATA_LOG atualizado incluindo | temVisao na 10ª posição (parts[9])
                    Print("DATA_LOG | " + GetGame().GetTime() + " | " + pBody.GetID() + " | " + pos[0] + " | " + pos[1] + " | " + pos[2] + " | " + dir[0] + " | " + dir[2] + " | " + isAiming + " | " + weaponName + " | " + temVisao);
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